// polar_ray_sampling_fast_fixed.ijm
// Fast, memory-backed ray sampling (no window switching in inner loops).
// Adds a safety check and progress logging.
// Requirements: draw ROIs and press T to add them to ROI Manager. Active image = original intensity image.
// Outputs to ~/IJ_cell_profiles/

angBins = 360;       
radialBinSize = 1.0;  
maxAllowedSamples = 5000000; 

outDir = getDirectory("home") + "IJ_cell_profiles/";
File.makeDirectory(outDir);
run("Set Measurements...", "centroid redirect=None decimal=3");

roiCount = roiManager("count");
if (roiCount == 0) { showMessage("Error","No ROIs in ROI Manager. Draw ROIs and press T."); exit(); }

// Estimate cost (rough)
getDimensions(imgW, imgH, c, z, t);
approx_maxR = floor(sqrt((imgW/2)*(imgW/2) + (imgH/2)*(imgH/2)));
approx_rBins = floor(approx_maxR / radialBinSize) + 1;
est_total = approx_rBins * angBins * roiCount;
if (est_total > maxAllowedSamples) {
    showMessage("Too big", "Estimated samples = " + est_total + "\nReduce angBins or increase radialBinSize and try again.");
    exit();
}

for (i = 0; i < roiCount; i++) {

    // ensure ROI active and restore selection
    roiManager("Select", i);
    run("Restore Selection");
    getSelectionBounds(x0, y0, w, h);
    if (w == 0 || h == 0) { print("Skipping ROI " + i + " (zero bounds)."); continue; }

    // duplicate full image and crop to ROI bounds
    tmpTitle = "tmp_int_" + i;
    run("Duplicate...", "title=" + tmpTitle);
    selectWindow(tmpTitle);
    roiManager("Select", i);
    run("Restore Selection");
    run("Clear Outside");
    run("Crop");
    run("32-bit");

    // duplicate mask aligned to crop
    run("Duplicate...", "title=" + "tmp_mask_" + i);
    selectWindow("tmp_mask_" + i);
    run("8-bit");
    setThreshold(1, 255);
    run("Convert to Mask");

    // get dimensions and centroid mapping
    selectWindow(tmpTitle);
    getDimensions(W, H, ch, sl, fr);
    run("Measure");
    xcent = getResult("X",0); ycent = getResult("Y",0);
    run("Clear Results");
    xc = xcent - x0; yc = ycent - y0;
    if (xc < 0 || xc > W) xc = W/2;
    if (yc < 0 || yc > H) yc = H/2;

    // compute rBins for this ROI
    if (xc > (W - xc)) dxmax = xc; else dxmax = W - xc;
    if (yc > (H - yc)) dymax = yc; else dymax = H - yc;
    maxR = floor(sqrt(dxmax*dxmax + dymax*dymax));
    rBins = floor(maxR / radialBinSize) + 1;
    if (rBins < 1) rBins = 1;

    // safety check (per-ROI)
    samples_here = rBins * angBins;
    if (samples_here > maxAllowedSamples) {
        selectWindow(tmpTitle); close();
        selectWindow("tmp_mask_" + i); close();
        showMessage("ROI too big", "ROI " + i + " would sample " + samples_here + " points. Reduce angBins or increase radialBinSize.");
        exit();
    }

    // --- Load intensity and mask into arrays (fast memory access) ---
    intensityArr = newArray(W * H);
    maskArr = newArray(W * H);

    selectWindow(tmpTitle);
    for (yy = 0; yy < H; yy++) {
        for (xx = 0; xx < W; xx++) {
            intensityArr[yy*W + xx] = getPixel(xx, yy);
        }
    }

    selectWindow("tmp_mask_" + i);
    for (yy = 0; yy < H; yy++) {
        for (xx = 0; xx < W; xx++) {
            maskArr[yy*W + xx] = getPixel(xx, yy); // 0 or 255
        }
    }

    // Precompute trig
    cosA = newArray(angBins);
    sinA = newArray(angBins);
    angCenters = newArray(angBins);
    for (a=0; a<angBins; a++) {
        angCenters[a] = (a + 0.5) * (360.0 / angBins);
        theta = angCenters[a] * PI / 180.0;
        cosA[a] = cos(theta);
        sinA[a] = sin(theta);
    }

    // Prepare accumulators
    totalBins = rBins * angBins;
    sumArr = newArray(totalBins);
    cntArr = newArray(totalBins);
    maxRB = newArray(angBins);
    for (a=0; a<angBins; a++) maxRB[a] = -1;

    // Bilinear interpolation helper (works on memory arrays)
    function bilinear_mem(xf, yf) {
        x0i = floor(xf); y0i = floor(yf);
        x1i = x0i + 1; y1i = y0i + 1;
        if (x0i < 0 || y0i < 0 || x1i >= W || y1i >= H) return 0;
        dx = xf - x0i; dy = yf - y0i;
        v00 = intensityArr[y0i*W + x0i];
        v10 = intensityArr[y0i*W + x1i];
        v01 = intensityArr[y1i*W + x0i];
        v11 = intensityArr[y1i*W + x1i];
        return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11;
    }

    // --- Ray sampling (use arrays only; no selectWindow inside loops) ---
    progressInterval = floor(angBins / 10);
    if (progressInterval < 1) progressInterval = 1;

    for (a = 0; a < angBins; a++) {
        for (rb = 0; rb < rBins; rb++) {
            rcenter = (rb + 0.5) * radialBinSize;
            xr = xc + rcenter * cosA[a];
            yr = yc + rcenter * sinA[a];
            xi = floor(xr + 0.5); yi = floor(yr + 0.5); // nearest pixel for mask test
            if (xi < 0 || yi < 0 || xi >= W || yi >= H) continue;
            if (maskArr[yi*W + xi] == 0) continue; // outside mask
            // inside: bilinear sample intensity
            val = bilinear_mem(xr, yr);
            idx = rb*angBins + a;
            sumArr[idx] = sumArr[idx] + val;
            cntArr[idx] = cntArr[idx] + 1;
            if (rb > maxRB[a]) maxRB[a] = rb;
        }
        if ((a % progressInterval) == 0) print("ROI " + i + " progress: angle " + a + " / " + angBins);
    }

    print("ROI " + i + " done. crop " + W + "x" + H + " rBins=" + rBins + " angBins=" + angBins);

    // Build clipped CSVs
    header = "r_px";
    for (a=0; a<angBins; a++) header = header + "," + angCenters[a];
    header = header + "\n";
    matCSV = header; countsCSV = header;

    for (rb=0; rb<rBins; rb++) {
        center_r = (rb + 0.5) * radialBinSize;
        line = "" + center_r;
        cntline = "" + center_r;
        for (a=0; a<angBins; a++) {
            idx = rb*angBins + a;
            if (maxRB[a] >= 0 && rb <= maxRB[a]) {
                if (cntArr[idx] > 0) meanV = sumArr[idx] / cntArr[idx]; else meanV = 0;
                line = line + "," + meanV;
                cntline = cntline + "," + cntArr[idx];
            } else {
                line = line + ",NaN";
                cntline = cntline + ",0";
            }
        }
        matCSV = matCSV + line + "\n";
        countsCSV = countsCSV + cntline + "\n";
    }

    File.saveString(matCSV, outDir + "cell_" + i + "_polar_clipped_ray_fast.csv");
    File.saveString(countsCSV, outDir + "cell_" + i + "_polar_counts_clipped_ray_fast.csv");

    // Save heatmap TIFF (theta x r)
    newImage("polar_fast_"+i, "32-bit", angBins, rBins, 1);
    for (rb=0; rb<rBins; rb++) for (a=0; a<angBins; a++) {
        idx = rb*angBins + a;
        if (maxRB[a] >= 0 && rb <= maxRB[a] && cntArr[idx] > 0)
            setPixel(a, rb, sumArr[idx] / cntArr[idx]);
        else
            setPixel(a, rb, 0);
    }
    saveAs("Tiff", outDir + "cell_" + i + "_polar_clipped_ray_fast.tif");
    close();

    // cleanup temp windows
    selectWindow(tmpTitle); close();
    selectWindow("tmp_mask_" + i); close();
}

showMessage("Done", "Fast ray-sampled polar CSVs saved to:\n" + outDir);
