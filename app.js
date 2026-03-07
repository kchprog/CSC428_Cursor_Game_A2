// =====================================================================
// app.js — Cursor Technique Study
// =====================================================================
//
// README / OVERVIEW
// ─────────────────
// Full within-subjects cursor study: 3 (Technique) × 3 (Movement) × 3 (Clustering).
//
// TECHNIQUE:  BUBBLE | POINT | AREA
// MOVEMENT:   STATIC | SLOW (linear ~30 px/s) | FAST (Catmull-Rom spline ~80 px/s)
// CLUSTERING: LOW (20 targets) | MEDIUM (40 targets) | HIGH (80 targets, clustered)
//
// SEQUENCE GENERATION
//   Technique block order: Latin square keyed on (participantNumber % 3).
//     Row 0 → BUBBLE → POINT  → AREA
//     Row 1 → POINT  → AREA   → BUBBLE
//     Row 2 → AREA   → BUBBLE → POINT
//   Within each technique block: the 9 (movement × clustering) conditions are
//   independently re-randomised per participant and per block.
//   Each condition: TRIALS_PER_CONDITION trials using the same target set.
//   Total recorded trials: 3 × 9 × TRIALS_PER_CONDITION = 270 (default).
//
// FAMILIARIZATION
//   Optional pre-study practice phase (default: FAM_TRIALS_DEFAULT = 15 trials).
//   Each trial draws a random (technique × movement × clustering) combination.
//   No data is logged. A persistent banner labels all familiarization activity.
//   Participants may reduce the count to 0 in the setup form to skip entirely.
//
// ACTIVE LOGGING
//   • After every trial: full session state written to localStorage (crash backup).
//   • After every technique block (~90 trials): JSON file automatically downloaded.
//   • On session completion: full session JSON + flat trajectory JSON downloaded.
//   • A "Save Now" button is available throughout the study for manual backup.
//
// OUTPUT FILES
//   P{N}_block{B}_{TECHNIQUE}.json  intermediate save per technique block
//   P{N}_session.json               complete session on completion
//   P{N}_trajectories.json          flat trial-level closure + trajectory log
//
// SESSION JSON SCHEMA
//   sessionStart / sessionEnd    ISO timestamps
//   participant                  integer
//   latinSquareRow               participant % 3
//   sequenceOrder                [technique, technique, technique] — actual order
//   conditions[]                 27 condition objects
//     .technique / .movement / .clustering
//     .conditionIndex            0–26
//     .techBlock                 1–3
//     .condInBlock               1–9  (within-block randomised position)
//     .conditionStart / End      ISO timestamps
//     .conditionStats            aggregate stats (see STATS SCHEMA below)
//     .trials[]                  TRIALS_PER_CONDITION trial objects
//   sessionStats                 aggregate over all 270 trials
//
// TRIAL SCHEMA
//   trialNumber            1-based within condition
//   time_ms                onset-to-click duration
//   errorClickCount        non-target clicks before acquisition
//   clickX / clickY        click coordinates (px)
//   targetX / targetY      target centroid at moment of click (px)
//   targetRadius           target radius (px)
//   distanceToCenter_px    Euclidean click-to-centroid distance
//   normalizedDistance     distanceToCenter / radius
//   clickedInsideTarget    boolean
//   amplitude_px           distance from previous target centroid (null: trial 1)
//   prevTargetX / Y        previous target centroid (null: trial 1)
//   targetVx / Vy_px_per_s instantaneous velocity at click moment (finite diff)
//   targetSpeed_px_per_s   velocity magnitude
//   closureCurve           [{t_ms, dist_px}] sampled every ~10 ms
//   normalizedTrajectory   [{t_ms, x, y, dist_px}] target-centred rotated frame;
//                          null for trial 1 of each condition
//
// STATS SCHEMA
//   avgTime_ms, medianTime_ms, stdDevTime_ms
//   totalErrorClicks, avgErrorClicks
//   avgDistanceFromCenter_px, medianDistanceFromCenter_px
//   avgNormalizedDistance, medianNormalizedDistance
//   clicksInsideTarget, clicksOutsideTarget, precisionRate_percent
//   shannonEligibleTrials, shannonAvgID_bits
//   throughput_shannon_bps, throughput_hoffmann_bps
// =====================================================================

"use strict";

// =============================================================================
// CONSTANTS
// =============================================================================

var W = 600, H = 600;
var MIN_RADIUS = 10, MAX_RADIUS = 30;
var SPD_SLOW = 30, SPD_FAST = 80;
var AREA_RADIUS = 50;
var TRIALS_PER_CONDITION = 20;
var FAM_TRIALS_DEFAULT   = 15;
var LS_KEY = "cursorStudy_backup";

// Latin square: row index = participantNumber % 3
var LATIN_SQUARE = [
    ["BUBBLE", "POINT",  "AREA"  ],   // row 0
    ["POINT",  "AREA",   "BUBBLE"],   // row 1
    ["AREA",   "BUBBLE", "POINT" ]    // row 2
];

var CLUSTER_PARAMS = {
    LOW:    { numTargets: 10, minSep: 50 },
    MEDIUM: { numTargets: 40, minSep: 20 },
    HIGH:   { numTargets: 80, minSep: 5  }
};

// =============================================================================
// STATE MACHINE
// =============================================================================
//
//  SETUP       setup form visible
//  FAM_WAIT    between familiarisation trials (or before the first); click to proceed
//  FAM_TRIAL   familiarisation trial active — no data logged
//  FAM_DONE    all fam trials complete; click to begin recorded study
//  COND_WAIT   interstitial before a condition starts; click to proceed
//  TRIAL       main study trial active — data logged
//  BLOCK_REST  technique-block boundary; data saved; click to continue
//  DONE        session finished
//
var appState = "SETUP";

// =============================================================================
// STATE VARIABLES
// =============================================================================

var svg = null;

// Session metadata
var participant;
var famTrialsTotal;
var famTrialsDone = 0;

// Sequence
var sessionSequence   = [];
var conditionIndex    = 0;
var trialsInCondition = 0;

// Active condition
var currentTechnique  = "POINT";
var currentMovement   = "STATIC";
var currentClustering = "MEDIUM";
var numTargets        = 40;
var minSep            = 20;

// Trial state
var targets          = [];
var clickTarget      = -1;
var prevTargetPos    = null;
var trialStartTime   = 0;
var trialErrorClicks = 0;

// Animation
var animFrame     = null;
var lastFrameTime = null;

// Mouse tracking
var currentMousePos = [0, 0];

// Trajectory sampling
var sampleInterval = null;
var trialSamples   = [];

// Data structures
var sessionData     = null;
var currentCondData = null;
var trajectoryLog   = [];

// =============================================================================
// UTILITIES
// =============================================================================

function distance(a, b) {
    var dx = b[0] - a[0], dy = b[1] - a[1];
    return Math.sqrt(dx * dx + dy * dy);
}

function shuffle(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
}

function r2(v) { return Math.round(v * 100)   / 100;  }
function r3(v) { return Math.round(v * 1000)  / 1000; }

function arrayMean(arr) {
    if (!arr.length) return 0;
    return arr.reduce(function(a, b) { return a + b; }, 0) / arr.length;
}

function arraySampleStdDev(arr) {
    if (arr.length < 2) return 0;
    var m = arrayMean(arr);
    return Math.sqrt(
        arr.reduce(function(s, v) { return s + Math.pow(v - m, 2); }, 0) /
        (arr.length - 1)
    );
}

function median(arr) {
    if (!arr.length) return 0;
    var s = arr.slice().sort(function(a, b) { return a - b; });
    var m = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

// =============================================================================
// SEQUENCE GENERATION
// =============================================================================

function buildSessionSequence(pNum) {
    var row      = pNum % 3;
    var techOrder = LATIN_SQUARE[row].slice();

    var moves = ["STATIC", "SLOW", "FAST"];
    var dens  = ["LOW", "MEDIUM", "HIGH"];
    var pairs = [];
    for (var mi = 0; mi < 3; mi++)
        for (var di = 0; di < 3; di++)
            pairs.push([moves[mi], dens[di]]);

    var seq = [];
    for (var ti = 0; ti < 3; ti++) {
        var shuffled = shuffle(pairs.slice());
        for (var pi = 0; pi < 9; pi++) {
            seq.push({
                technique:   techOrder[ti],
                movement:    shuffled[pi][0],
                clustering:  shuffled[pi][1],
                techBlock:   ti + 1,
                condInBlock: pi + 1
            });
        }
    }
    return seq;
}

function randomCondition() {
    var techs = ["BUBBLE", "POINT", "AREA"];
    var moves = ["STATIC", "SLOW", "FAST"];
    var dens  = ["LOW", "MEDIUM", "HIGH"];
    return {
        technique:  techs[Math.floor(Math.random() * 3)],
        movement:   moves[Math.floor(Math.random() * 3)],
        clustering: dens [Math.floor(Math.random() * 3)]
    };
}

function applyCondition(cond) {
    currentTechnique  = cond.technique;
    currentMovement   = cond.movement;
    currentClustering = cond.clustering;
    var p = CLUSTER_PARAMS[currentClustering];
    numTargets = p.numTargets;
    minSep     = p.minSep;
}

// =============================================================================
// CATMULL-ROM SPLINE  (FAST movement)
// =============================================================================

function catmullRomPos(p0, p1, p2, p3, t) {
    var t2 = t * t, t3 = t2 * t;
    return [
        0.5 * (2*p1[0] + (-p0[0]+p2[0])*t
             + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2
             + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
        0.5 * (2*p1[1] + (-p0[1]+p2[1])*t
             + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2
             + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    ];
}

function catmullRomSegLen(p0, p1, p2, p3) {
    var samples = 12, len = 0;
    var prev = catmullRomPos(p0, p1, p2, p3, 0);
    for (var k = 1; k <= samples; k++) {
        var curr = catmullRomPos(p0, p1, p2, p3, k / samples);
        var dx = curr[0] - prev[0], dy = curr[1] - prev[1];
        len += Math.sqrt(dx * dx + dy * dy);
        prev = curr;
    }
    return Math.max(len, 1);
}

function clampToBounds(pt, margin) {
    return [
        Math.max(margin, Math.min(W - margin, pt[0])),
        Math.max(margin, Math.min(H - margin, pt[1]))
    ];
}

function initSplineState(startPos, rad) {
    var margin  = rad + 10;
    var step    = 100 + Math.random() * 120;
    var heading = Math.random() * Math.PI * 2;
    var p0 = clampToBounds(
        [startPos[0] - Math.cos(heading) * step,
         startPos[1] - Math.sin(heading) * step], margin);
    var p1 = [startPos[0], startPos[1]];
    var turn1 = heading + (Math.random() - 0.5) * Math.PI * 2;
    var step2 = 100 + Math.random() * 120;
    var p2 = clampToBounds(
        [p1[0] + Math.cos(turn1) * step2,
         p1[1] + Math.sin(turn1) * step2], margin);
    var turn2 = turn1 + (Math.random() - 0.5) * Math.PI * 2;
    var step3 = 100 + Math.random() * 120;
    var p3 = clampToBounds(
        [p2[0] + Math.cos(turn2) * step3,
         p2[1] + Math.sin(turn2) * step3], margin);
    return { pts: [p0, p1, p2, p3], t: 0, segLen: catmullRomSegLen(p0, p1, p2, p3) };
}

function advanceSplineTarget(target, dt) {
    var ss     = target[3];
    var rad    = target[1];
    var margin = rad + 6;
    var pts    = ss.pts;
    var prevX  = target[0][0], prevY = target[0][1];

    ss.t += (SPD_FAST * dt) / ss.segLen;

    var guard = 0;
    while (ss.t >= 1.0 && guard < 20) {
        guard++;
        ss.t -= 1.0;
        pts.shift();
        var last    = pts[pts.length - 1];
        var prev2   = pts[pts.length - 2];
        var bearing = Math.atan2(last[1] - prev2[1], last[0] - prev2[0]);
        var turn    = bearing + (Math.random() - 0.5) * 2.44;
        var step    = 100 + Math.random() * 120;
        var newPt   = clampToBounds(
            [last[0] + Math.cos(turn) * step,
             last[1] + Math.sin(turn) * step], margin);
        pts.push(newPt);
        ss.segLen = catmullRomSegLen(pts[0], pts[1], pts[2], pts[3]);
    }

    var pos = clampToBounds(catmullRomPos(pts[0], pts[1], pts[2], pts[3], ss.t), margin);
    target[0][0] = pos[0];
    target[0][1] = pos[1];
    if (dt > 0) {
        target[2][0] = (pos[0] - prevX) / dt;
        target[2][1] = (pos[1] - prevY) / dt;
    }
}

// =============================================================================
// TARGET INITIALISATION
// =============================================================================

function initTargets() {
    var radRange = MAX_RADIUS - MIN_RADIUS;
    var minX = MAX_RADIUS + 10, maxX = W - MAX_RADIUS - 10;
    var minY = MAX_RADIUS + 10, maxY = H - MAX_RADIUS - 10;
    var xRange = maxX - minX, yRange = maxY - minY;
    var result = [];
    var maxAttempts   = currentClustering === "HIGH" ? 3000 : 600;
    var clusterCenters = null;
    var clusterSpread  = 100;

    if (currentClustering === "HIGH") {
        var numClusters = 8, minClusterSep = 180;
        clusterCenters = [];
        var ca = 0;
        while (clusterCenters.length < numClusters && ca < 1000) {
            ca++;
            var cx = Math.random() * xRange + minX;
            var cy = Math.random() * yRange + minY;
            var ok = true;
            for (var k = 0; k < clusterCenters.length && ok; k++) {
                var dx = cx - clusterCenters[k][0], dy = cy - clusterCenters[k][1];
                if (Math.sqrt(dx*dx + dy*dy) < minClusterSep) ok = false;
            }
            if (ok) clusterCenters.push([cx, cy]);
        }
        while (clusterCenters.length < 3)
            clusterCenters.push([Math.random()*xRange+minX, Math.random()*yRange+minY]);
    }

    function samplePt() {
        if (clusterCenters) {
            var c   = clusterCenters[Math.floor(Math.random() * clusterCenters.length)];
            var ang = Math.random() * Math.PI * 2;
            var d   = Math.random() * clusterSpread;
            return [
                Math.max(minX, Math.min(maxX, c[0] + Math.cos(ang) * d)),
                Math.max(minY, Math.min(maxY, c[1] + Math.sin(ang) * d))
            ];
        }
        return [Math.random() * xRange + minX, Math.random() * yRange + minY];
    }

    for (var i = 0; i < numTargets; i++) {
        var collision = true, attempts = 0, pt, rad;
        while (collision && attempts < maxAttempts) {
            attempts++;
            pt  = samplePt();
            rad = Math.random() * radRange + MIN_RADIUS;
            collision = false;
            for (var j = 0; j < result.length && !collision; j++) {
                var sep = distance(pt, result[j][0]);
                var minAllowed = currentClustering === "HIGH"
                    ? rad + result[j][1] - Math.min(rad, result[j][1]) / 2
                    : rad + result[j][1] + minSep;
                if (sep < minAllowed) collision = true;
            }
        }
        if (!pt) { pt = samplePt(); rad = Math.random() * radRange + MIN_RADIUS; }

        var spd = currentMovement === "SLOW" ? SPD_SLOW
                : currentMovement === "FAST" ? SPD_FAST : 0;
        var a   = Math.random() * Math.PI * 2;
        var ss  = currentMovement === "FAST" ? initSplineState(pt, rad) : null;
        result.push([pt, rad, [Math.cos(a) * spd, Math.sin(a) * spd], ss]);
    }
    return result;
}

// =============================================================================
// RENDERING
// =============================================================================

function renderTargets() {
    svg.selectAll(".targetCircles").remove();
    svg.selectAll(".targetCircles").data(targets).enter().append("circle")
        .attr("class", "targetCircles")
        .attr("cx",    function(d) { return d[0][0]; })
        .attr("cy",    function(d) { return d[0][1]; })
        .attr("r",     function(d) { return d[1] - 1; })
        .attr("stroke-width", 2).attr("stroke", "limegreen")
        .attr("fill", "transparent");   // ← was "white"
    updateTargetsFill(-1, clickTarget);
}

function updateTargetsFill(capturedIdx, targetIdx) {
    svg.selectAll(".targetCircles")
        .attr("fill", function(d, i) {
            if (i === targetIdx && i === capturedIdx) return "darkred";
            if (i === targetIdx)   return "magenta";
            if (i === capturedIdx) return "limegreen";
            return "none";
        })
        .attr("stroke", function(d, i) {
            if (i === targetIdx && i === capturedIdx) return "darkred";
            if (i === targetIdx)   return "limegreen";
            if (i === capturedIdx) return "limegreen";
            return "limegreen";
        });
}

function hideCursors() {
    svg.select(".cursorCircle").attr("r", 0);
    svg.select(".cursorMorphCircle").attr("cx", 0).attr("cy", 0).attr("r", 0);
}

// =============================================================================
// CURSOR TECHNIQUES
// =============================================================================

function getTargetCapturedByBubbleCursor(mouse) {
    if (!targets.length) {
        svg.select(".cursorCircle").attr("cx", mouse[0]).attr("cy", mouse[1]).attr("r", 0);
        svg.select(".cursorMorphCircle").attr("cx", 0).attr("cy", 0).attr("r", 0);
        return -1;
    }
    var containDists = [], intersectDists = [];
    var currMinIdx = 0;
    for (var idx = 0; idx < targets.length; idx++) {
        var d = distance(mouse, targets[idx][0]), r = targets[idx][1];
        containDists.push(d + r);
        intersectDists.push(d - r);
        if (intersectDists[idx] < intersectDists[currMinIdx]) currMinIdx = idx;
    }
    var secondMinIdx = (currMinIdx + 1) % targets.length;
    for (var idx = 0; idx < targets.length; idx++) {
        if (idx !== currMinIdx && intersectDists[idx] < intersectDists[secondMinIdx])
            secondMinIdx = idx;
    }
    var cursorRadius = Math.min(containDists[currMinIdx], intersectDists[secondMinIdx]);
    svg.select(".cursorCircle").attr("cx", mouse[0]).attr("cy", mouse[1]).attr("r", cursorRadius);
    if (cursorRadius < containDists[currMinIdx]) {
        svg.select(".cursorMorphCircle")
            .attr("cx", targets[currMinIdx][0][0])
            .attr("cy", targets[currMinIdx][0][1])
            .attr("r",  targets[currMinIdx][1] + 5);
    } else {
        svg.select(".cursorMorphCircle").attr("cx", 0).attr("cy", 0).attr("r", 0);
    }
    return currMinIdx;
}

function getTargetCapturedByPointCursor(mouse) {
    var captured = -1;
    for (var idx = 0; idx < targets.length; idx++) {
        if (distance(mouse, targets[idx][0]) <= targets[idx][1]) captured = idx;
    }
    svg.select(".cursorCircle").attr("cx", mouse[0]).attr("cy", mouse[1]).attr("r", 0);
    svg.select(".cursorMorphCircle").attr("cx", 0).attr("cy", 0).attr("r", 0);
    return captured;
}

function getTargetCapturedByAreaCursor(mouse) {
    var capturedArea = -1, capturedPoint = -1, numCaptured = 0;
    for (var idx = 0; idx < targets.length; idx++) {
        var d = distance(mouse, targets[idx][0]), r = targets[idx][1];
        if (d <= r + AREA_RADIUS) { capturedArea = idx; numCaptured++; }
        if (d <= r)                capturedPoint = idx;
    }
    var captured = capturedPoint > -1   ? capturedPoint
                 : numCaptured === 1     ? capturedArea : -1;
    svg.select(".cursorCircle")
        .attr("cx", mouse[0]).attr("cy", mouse[1])
        .attr("r", AREA_RADIUS).attr("fill", "lightgray");
    svg.select(".cursorMorphCircle").attr("cx", 0).attr("cy", 0).attr("r", 0);
    return captured;
}

function getCapturedTarget(mousePos) {
    if (currentTechnique === "BUBBLE") return getTargetCapturedByBubbleCursor(mousePos);
    if (currentTechnique === "POINT")  return getTargetCapturedByPointCursor(mousePos);
    if (currentTechnique === "AREA")   return getTargetCapturedByAreaCursor(mousePos);
    return -1;
}

// =============================================================================
// STATISTICS
// =============================================================================

function shannonEligible(trials) {
    return trials.filter(function(t) {
        return t.amplitude_px !== null && t.amplitude_px > 0 && t.prevTargetX !== null;
    });
}

function computeShannonThroughput(trials) {
    if (!trials.length) return { avgID: 0, throughput: 0 };
    var tps = [], ids = [];
    for (var i = 0; i < trials.length; i++) {
        var t  = trials[i];
        var W_ = 2 * t.targetRadius, MT = t.time_ms / 1000;
        if (W_ <= 0 || MT <= 0 || t.amplitude_px <= 0) continue;
        var ID = Math.log2(t.amplitude_px / W_ + 1);
        ids.push(ID); tps.push(ID / MT);
    }
    return { avgID: r3(arrayMean(ids)), throughput: r2(arrayMean(tps)) };
}

function computeHoffmannThroughput(trials) {
    if (!trials.length) return 0;
    var htps = [];
    for (var i = 0; i < trials.length; i++) {
        var t  = trials[i];
        var MT = t.time_ms / 1000, W_ = 2 * t.targetRadius;
        var dx = t.targetX - t.prevTargetX, dy = t.targetY - t.prevTargetY;
        var len = Math.sqrt(dx*dx + dy*dy);
        var ux  = len > 0 ? dx / len : 1, uy = len > 0 ? dy / len : 0;
        var vt  = Math.abs(t.targetVx_px_per_s * ux + t.targetVy_px_per_s * uy);
        var W_eff = Math.max(W_ - vt * MT, W_ * 0.1);
        htps.push(Math.log2(t.amplitude_px / W_eff + 1) / MT);
    }
    return arrayMean(htps);
}

function computeConditionStats(condData) {
    var trials = condData.trials;
    var times = [], dists = [], normDists = [], errors = [];
    var inside = 0, outside = 0;
    var eligible = shannonEligible(trials);

    for (var i = 0; i < trials.length; i++) {
        var t = trials[i];
        times.push(t.time_ms);
        errors.push(t.errorClickCount);
        if (t.distanceToCenter_px !== null) dists.push(t.distanceToCenter_px);
        if (t.normalizedDistance  !== null) normDists.push(t.normalizedDistance);
        if (t.clickedInsideTarget) inside++; else outside++;
    }

    var sh    = computeShannonThroughput(eligible);
    var total = inside + outside;

    return {
        totalTrials:                   trials.length,
        avgTime_ms:                    r2(arrayMean(times)),
        medianTime_ms:                 r2(median(times)),
        stdDevTime_ms:                 r2(arraySampleStdDev(times)),
        totalErrorClicks:              errors.reduce(function(a,b){return a+b;}, 0),
        avgErrorClicks:                r2(arrayMean(errors)),
        avgDistanceFromCenter_px:      r2(arrayMean(dists)),
        medianDistanceFromCenter_px:   r2(median(dists)),
        avgNormalizedDistance:         r3(arrayMean(normDists)),
        medianNormalizedDistance:      r3(median(normDists)),
        clicksInsideTarget:            inside,
        clicksOutsideTarget:           outside,
        precisionRate_percent:         r2(total > 0 ? (inside / total) * 100 : 0),
        shannonEligibleTrials:         eligible.length,
        shannonAvgID_bits:             sh.avgID,
        throughput_shannon_bps:        sh.throughput,
        throughput_hoffmann_bps:       r2(computeHoffmannThroughput(eligible))
    };
}

function computeSessionStats() {
    var allTrials = [];
    for (var c = 0; c < sessionData.conditions.length; c++)
        for (var i = 0; i < sessionData.conditions[c].trials.length; i++)
            allTrials.push(sessionData.conditions[c].trials[i]);

    var stats = computeConditionStats({ trials: allTrials });
    stats.totalConditions = sessionData.conditions.length;

    // Per-technique throughput summary
    var techTP = {};
    for (var c = 0; c < sessionData.conditions.length; c++) {
        var cond = sessionData.conditions[c];
        var tp   = cond.conditionStats.throughput_shannon_bps;
        if (!techTP[cond.technique]) techTP[cond.technique] = [];
        techTP[cond.technique].push(tp);
    }
    stats.avgThroughputByTechnique = {};
    for (var tech in techTP)
        stats.avgThroughputByTechnique[tech] = r2(arrayMean(techTP[tech]));

    return stats;
}

// =============================================================================
// CLOSURE / TRAJECTORY SAMPLING
// =============================================================================

function startSampling() {
    stopSampling();
    trialSamples = [];
    var t0 = new Date().getTime();
    sampleInterval = setInterval(function() {
        if (appState !== "TRIAL" && appState !== "FAM_TRIAL") return;
        if (clickTarget < 0 || clickTarget >= targets.length) return;
        var tgt = targets[clickTarget][0];
        trialSamples.push({
            t_ms:    new Date().getTime() - t0,
            cx:      +(currentMousePos[0].toFixed(1)),
            cy:      +(currentMousePos[1].toFixed(1)),
            tx:      +(tgt[0].toFixed(1)),
            ty:      +(tgt[1].toFixed(1)),
            dist_px: +(distance(currentMousePos, tgt).toFixed(1))
        });
    }, 10);
}

function stopSampling() {
    if (sampleInterval !== null) {
        clearInterval(sampleInterval);
        sampleInterval = null;
    }
}

// Rotate cursor path into a target-centred frame where +x points toward
// prevTarget, so all trajectories "arrive from the right" regardless of
// on-screen orientation. Returns null for the first trial of each condition.
function normalizeTrajectory(samples, targetPos, prevPos) {
    if (!prevPos || !samples.length) return null;
    var tx    = targetPos[0], ty = targetPos[1];
    var px    = prevPos[0] - tx, py = prevPos[1] - ty;
    var angle = Math.atan2(py, px);
    var cosA  = Math.cos(-angle), sinA = Math.sin(-angle);
    return samples.map(function(s) {
        var dx = s.cx - tx, dy = s.cy - ty;
        return {
            t_ms:    s.t_ms,
            x:       +((dx * cosA - dy * sinA).toFixed(2)),
            y:       +((dx * sinA + dy * cosA).toFixed(2)),
            dist_px: s.dist_px
        };
    });
}

// =============================================================================
// DATA LOGGING
// =============================================================================

function logTrialData(trialTime, clickPos, targetPos, targetRad,
                      amplitude, prevX, prevY, vx, vy,
                      errorCount, closureCurve, normTraj) {
    var dtc   = distance(clickPos, targetPos);
    var speed = Math.sqrt(vx * vx + vy * vy);

    var trialData = {
        trialNumber:           trialsInCondition,
        time_ms:               trialTime,
        errorClickCount:       errorCount,
        clickX:                +(clickPos[0].toFixed(1)),
        clickY:                +(clickPos[1].toFixed(1)),
        targetX:               +(targetPos[0].toFixed(1)),
        targetY:               +(targetPos[1].toFixed(1)),
        targetRadius:          targetRad,
        distanceToCenter_px:   +(dtc.toFixed(2)),
        normalizedDistance:    targetRad > 0 ? +(dtc / targetRad).toFixed(3) : null,
        clickedInsideTarget:   dtc <= targetRad,
        amplitude_px:          amplitude !== null ? +(amplitude.toFixed(2)) : null,
        prevTargetX:           prevX !== null ? +(prevX.toFixed(1)) : null,
        prevTargetY:           prevY !== null ? +(prevY.toFixed(1)) : null,
        targetVx_px_per_s:     +(vx.toFixed(2)),
        targetVy_px_per_s:     +(vy.toFixed(2)),
        targetSpeed_px_per_s:  +(speed.toFixed(2)),
        closureCurve:          closureCurve,
        normalizedTrajectory:  normTraj
    };

    currentCondData.trials.push(trialData);

    trajectoryLog.push({
        participant:          participant,
        conditionIndex:       conditionIndex,
        technique:            currentTechnique,
        movement:             currentMovement,
        clustering:           currentClustering,
        trial:                trialsInCondition,
        targetRadius_px:      targetRad,
        amplitude_px:         trialData.amplitude_px,
        time_ms:              trialTime,
        errorClickCount:      errorCount,
        closureCurve:         closureCurve,
        normalizedTrajectory: normTraj
    });

    // Write crash-recovery backup after every trial
    saveToLocalStorage();
}

function saveToLocalStorage() {
    try {
        localStorage.setItem(LS_KEY, JSON.stringify({
            savedAt:        new Date().toISOString(),
            conditionIndex: conditionIndex,
            sessionData:    sessionData,
            currentCond:    currentCondData
        }));
    } catch (e) { /* quota exceeded or unavailable — silently skip */ }
}

// =============================================================================
// FILE SAVING
// =============================================================================

var SESSION_README = {
    "_README_study":      "Cursor Technique Study — 3×3×3 fully within-subjects factorial design.",
    "_README_ivs":        "IV1 Technique: BUBBLE|POINT|AREA. IV2 Movement: STATIC|SLOW|FAST. IV3 Clustering: LOW|MEDIUM|HIGH.",
    "_README_sequence":   "Technique block order: Latin square on participant%3. Within each block: 9 movement×clustering conditions in randomised order.",
    "_README_amplitude":  "amplitude_px and prevTarget: distance between successive target centroids, computed at moment of target assignment.",
    "_README_velocity":   "targetVx/Vy_px_per_s: finite-difference velocity at click moment. For FAST targets this reflects the Catmull-Rom spline tangent.",
    "_README_shannon":    "throughput_shannon_bps: ID=log2(A/W+1) where W=2r; TP=ID/MT averaged over eligible trials.",
    "_README_hoffmann":   "throughput_hoffmann_bps: ID=log2(A/W_eff+1)/MT where W_eff=W−|v_along|·MT, clamped to 0.1·W.",
    "_README_eligible":   "Trial 1 of each condition excluded from throughput calculations (no predecessor amplitude).",
    "_README_closure":    "closureCurve: [{t_ms, dist_px}] ~10 ms samples from onset to click. Target position is live for moving conditions.",
    "_README_trajectory": "normalizedTrajectory: [{t_ms, x, y, dist_px}] in target-centred frame rotated so +x points toward prevTarget. null for trial 1."
};

var TRAJECTORY_README = {
    "_README_format":      "Flat trial-level log. One object per completed trial across the full session.",
    "_README_closure":     "closureCurve: [{t_ms, dist_px}] ~10 ms samples. For moving targets the target position is updated live at each sample.",
    "_README_trajectory":  "normalizedTrajectory: [{t_ms, x, y, dist_px}] rotated frame. null for trial 1 of each condition.",
    "_README_aggregation": "Group by (technique, movement, clustering). Bin closureCurve on t_ms (nearest 10 ms) and average dist_px. " +
                           "Overlay normalizedTrajectory polylines at ~0.05 opacity for density visualisation."
};

function saveJSON(data, filename) {
    var blob = new Blob(
        [JSON.stringify(data, null, 2)],
        { type: "application/json;charset=utf-8;" }
    );
    saveAs(blob, filename);
}

// Called automatically at the end of each technique block (~90 trials).
function saveTechBlock(techName, blockNum) {
    var blockConditions = sessionData.conditions.filter(function(c) {
        return c.techBlock === blockNum;
    });
    var out = Object.assign({}, SESSION_README, {
        _note:             "INTERMEDIATE SAVE — technique block " + blockNum + " of 3",
        savedAt:           new Date().toISOString(),
        participant:       participant,
        techBlock:         blockNum,
        technique:         techName,
        conditionsInBlock: blockConditions
    });
    saveJSON(out, "P" + participant + "_block" + blockNum + "_" + techName + ".json");
}

// Called at session completion.
function saveFullSession() {
    sessionData.sessionEnd   = new Date().toISOString();
    sessionData.sessionStats = computeSessionStats();

    var sessionOut = Object.assign({}, SESSION_README, sessionData);
    saveJSON(sessionOut, "P" + participant + "_session.json");

    var trajOut = Object.assign({}, TRAJECTORY_README, { trials: trajectoryLog });
    saveJSON(trajOut, "P" + participant + "_trajectories.json");
}

// Manual save — can be triggered at any time via the "Save Now" button.
function saveNow() {
    if (!sessionData) return;
    var snapshot = Object.assign({}, SESSION_README, {
        _note:          "MANUAL PARTIAL SAVE — session not yet complete",
        savedAt:        new Date().toISOString(),
        participant:    participant,
        conditionsDone: conditionIndex,
        conditions:     sessionData.conditions.slice()
    });
    saveJSON(snapshot, "P" + participant + "_partial_" + Date.now() + ".json");
}

// =============================================================================
// SVG INITIALISATION
// =============================================================================

function initSVG() {
    svg = d3.select("#canvas").append("svg:svg").attr("width", W).attr("height", H);

    svg.append("rect").attr("class", "bg")
        .attr("width", W).attr("height", H)
        .attr("fill", "white").attr("stroke", "#333");

    // Status lines (bottom-left area)
    svg.append("text").attr("class", "statusA").attr("x", 20).attr("y", 22)
        .attr("font-size", "13px").text("");
    svg.append("text").attr("class", "statusB").attr("x", 20).attr("y", 40)
        .attr("font-size", "13px").text("");
    svg.append("text").attr("class", "statusC").attr("x", 20).attr("y", 58)
        .attr("font-size", "12px").attr("fill", "#666").text("");

    // Familiarisation banner along the bottom edge
    svg.append("text").attr("class", "famBanner")
        .attr("x", W / 2).attr("y", H - 12)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("fill", "#c06000")
        .attr("font-weight", "bold").text("");

    // Top-right: trial counter and condition label
    svg.append("text").attr("class", "trialCounter")
        .attr("x", W - 12).attr("y", 20)
        .attr("text-anchor", "end")
        .attr("font-size", "12px").attr("font-weight", "bold").text("");

    svg.append("text").attr("class", "condLabel")
        .attr("x", W - 12).attr("y", 36)
        .attr("text-anchor", "end")
        .attr("font-size", "11px").attr("fill", "#555").text("");

    // Cursor overlays
    svg.append("circle").attr("class", "cursorCircle")
        .attr("cx", 0).attr("cy", 0).attr("r", 0)
        .attr("fill", "lightgray").attr("fill-opacity", "0.5")
        .attr("pointer-events", "none");
    svg.append("circle").attr("class", "cursorMorphCircle")
        .attr("cx", 0).attr("cy", 0).attr("r", 0)
        .attr("fill", "lightgray").attr("fill-opacity", "0.3")
        .attr("pointer-events", "none");

    svg.on("mousemove", onMouseMove);
    svg.on("click",     onClick);
}

function setStatus(a, b, c) {
    svg.select(".statusA").text(a || "");
    svg.select(".statusB").text(b || "");
    svg.select(".statusC").text(c || "");
}

function setTopRight(counter, label) {
    svg.select(".trialCounter").text(counter || "");
    svg.select(".condLabel").text(label    || "");
}

function setFamBanner(txt) {
    svg.select(".famBanner").text(txt || "");
}

// =============================================================================
// STATE TRANSITIONS — FAMILIARISATION
// =============================================================================

function goFamWait() {
    appState = "FAM_WAIT";
    stopAnimation();
    setFamBanner("⚑  FAMILIARIZATION — results not recorded");
    if (famTrialsDone === 0) {
        setStatus(
            "Familiarization: " + famTrialsTotal + " practice trials",
            "Click anywhere to begin. Conditions are randomised.",
            "Get comfortable with the task before the recorded study starts."
        );
    } else {
        setStatus(
            "Trial done.  " + (famTrialsTotal - famTrialsDone) + " familiarization trial(s) remaining.",
            "Click anywhere to continue.", "");
    }
    setTopRight("FAM " + famTrialsDone + "/" + famTrialsTotal, "");
}

function startFamTrial() {
    appState = "FAM_TRIAL";
    var cond = randomCondition();
    applyCondition(cond);

    setFamBanner(
        "⚑  FAM " + (famTrialsDone + 1) + "/" + famTrialsTotal +
        "  [" + currentTechnique + " · " + currentMovement + " · " + currentClustering + "]"
    );
    setStatus("", "", "");
    setTopRight(
        "FAM " + (famTrialsDone + 1) + "/" + famTrialsTotal,
        currentTechnique + " / " + currentMovement + " / " + currentClustering
    );

    prevTargetPos    = null;
    trialErrorClicks = 0;
    targets          = initTargets();
    clickTarget      = Math.floor(Math.random() * targets.length);
    renderTargets();
    trialStartTime   = new Date().getTime();
    startSampling();
    ensureAnimation();
}

function completeFamTrial() {
    stopSampling();
    svg.selectAll(".targetCircles").remove();
    hideCursors();
    stopAnimation();
    famTrialsDone++;

    if (famTrialsDone >= famTrialsTotal) {
        appState = "FAM_DONE";
        setFamBanner("");
        setTopRight("", "");
        var order = LATIN_SQUARE[participant % 3].join(" → ");
        setStatus(
            "Familiarization complete! (" + famTrialsTotal + " trials)",
            "Your technique block order: " + order,
            "Click anywhere to begin the recorded study."
        );
    } else {
        goFamWait();
    }
}

// =============================================================================
// STATE TRANSITIONS — MAIN STUDY
// =============================================================================

function goCondWait() {
    appState          = "COND_WAIT";
    trialsInCondition = 0;
    prevTargetPos     = null;

    if (conditionIndex >= sessionSequence.length) {
        goSessionDone();
        return;
    }

    var cond = sessionSequence[conditionIndex];
    applyCondition(cond);

    var isBlockStart = (cond.condInBlock === 1);
    var header = isBlockStart
        ? "▶  Technique Block " + cond.techBlock + "/3: " + cond.technique
        : "Condition " + (conditionIndex + 1) + "/27  —  Block " + cond.techBlock;

    setFamBanner("");
    setStatus(
        header,
        "Movement: " + cond.movement + "   Density: " + cond.clustering +
            "   (" + cond.condInBlock + "/9 in block)",
        "Click anywhere to begin (" + TRIALS_PER_CONDITION + " trials)"
    );
    setTopRight(
        "Cond " + (conditionIndex + 1) + "/27",
        cond.technique + " / " + cond.movement + " / " + cond.clustering
    );

    currentCondData = {
        conditionIndex: conditionIndex,
        technique:      cond.technique,
        movement:       cond.movement,
        clustering:     cond.clustering,
        techBlock:      cond.techBlock,
        condInBlock:    cond.condInBlock,
        conditionStart: null,
        conditionEnd:   null,
        conditionStats: null,
        trials:         []
    };
}

// Initialises the target set once for the whole condition and starts trial 1.
function startCondition() {
    currentCondData.conditionStart = new Date().toISOString();
    appState         = "TRIAL";
    prevTargetPos    = null;
    trialErrorClicks = 0;
    targets          = initTargets();
    clickTarget      = Math.floor(Math.random() * targets.length);
    renderTargets();
    setStatus("", "", "");
    setTopRight(
        "Trial 1/" + TRIALS_PER_CONDITION + "  Cond " + (conditionIndex + 1) + "/27",
        currentTechnique + " / " + currentMovement + " / " + currentClustering
    );
    trialStartTime = new Date().getTime();
    startSampling();
    ensureAnimation();
}

function onTrialSuccess(clickPos) {
    var trialTime = new Date().getTime() - trialStartTime;
    stopSampling();

    var closureCurve = trialSamples.map(function(s) {
        return { t_ms: s.t_ms, dist_px: s.dist_px };
    });
    var normTraj = normalizeTrajectory(
        trialSamples, targets[clickTarget][0], prevTargetPos
    );

    var tv        = targets[clickTarget][2];
    var amplitude = prevTargetPos
        ? distance(prevTargetPos, targets[clickTarget][0]) : null;
    var prevX     = prevTargetPos ? prevTargetPos[0] : null;
    var prevY     = prevTargetPos ? prevTargetPos[1] : null;

    trialsInCondition++;

    logTrialData(
        trialTime, clickPos,
        targets[clickTarget][0], targets[clickTarget][1],
        amplitude, prevX, prevY, tv[0], tv[1],
        trialErrorClicks, closureCurve, normTraj
    );

    prevTargetPos    = [targets[clickTarget][0][0], targets[clickTarget][0][1]];
    trialErrorClicks = 0;

    if (trialsInCondition >= TRIALS_PER_CONDITION) {
        completeCondition();
    } else {
        // Same target set — pick a new highlighted target
        var newTarget = clickTarget;
        while (newTarget === clickTarget && targets.length > 1)
            newTarget = Math.floor(Math.random() * targets.length);
        clickTarget = newTarget;
        updateTargetsFill(-1, clickTarget);
        setTopRight(
            "Trial " + (trialsInCondition + 1) + "/" + TRIALS_PER_CONDITION +
                "  Cond " + (conditionIndex + 1) + "/27",
            currentTechnique + " / " + currentMovement + " / " + currentClustering
        );
        trialStartTime = new Date().getTime();
        startSampling();
    }
}

function completeCondition() {
    currentCondData.conditionEnd   = new Date().toISOString();
    currentCondData.conditionStats = computeConditionStats(currentCondData);
    sessionData.conditions.push(currentCondData);
    currentCondData = null;
    saveToLocalStorage();

    svg.selectAll(".targetCircles").remove();
    hideCursors();
    stopAnimation();

    // Block boundary: every 9 conditions (indices 8, 17, 26)
    var isBlockEnd = (conditionIndex % 9 === 8);
    conditionIndex++;

    if (isBlockEnd) {
        var techBlockNum  = Math.floor(conditionIndex / 9);
        var techBlockName = LATIN_SQUARE[participant % 3][techBlockNum - 1];
        saveTechBlock(techBlockName, techBlockNum);

        if (conditionIndex >= sessionSequence.length) {
            goSessionDone();
        } else {
            appState = "BLOCK_REST";
            var nextTech = sessionSequence[conditionIndex].technique;
            setStatus(
                "Technique block " + techBlockNum + "/3 complete.  Data saved.",
                "Next block: " + nextTech,
                "Click anywhere to continue."
            );
            setTopRight("Block " + techBlockNum + "/3 done", "");
        }
    } else {
        goCondWait();
    }
}

function goSessionDone() {
    appState = "DONE";
    stopAnimation();
    stopSampling();
    saveFullSession();

    var total = sessionData.conditions.length * TRIALS_PER_CONDITION;
    setStatus(
        "Session complete!  All data saved.",
        "Thank you, Participant " + participant + ".",
        sessionData.conditions.length + " conditions, " + total + " trials recorded."
    );
    setTopRight("", "");
    setFamBanner("");

    // Show restart button
    d3.select("#saveNowBtn").style("display", "none");
    d3.select("#canvas").append("button")
        .style("position", "relative").style("display", "block")
        .style("margin", "12px 20px").style("padding", "8px 16px")
        .text("Start New Session")
        .on("click", function() {
            d3.select(this).remove();
            svg.remove(); svg = null;
            appState = "SETUP";
            d3.select("#saveNowBtn").style("display", "none");
            document.getElementById("setup").style.display = "block";
        });
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function onMouseMove() {
    currentMousePos = d3.mouse(this);
    if (appState === "TRIAL" || appState === "FAM_TRIAL") {
        updateTargetsFill(getCapturedTarget(currentMousePos), clickTarget);
    }
}

function onClick() {
    var mousePos = d3.mouse(this);

    switch (appState) {

        case "FAM_WAIT":
            startFamTrial();
            break;

        case "FAM_TRIAL": {
            var captured = getCapturedTarget(mousePos);
            if (captured !== clickTarget) {
                trialErrorClicks++;
                updateTargetsFill(captured, clickTarget);
            } else {
                completeFamTrial();
            }
            break;
        }

        case "FAM_DONE":
            goCondWait();
            break;

        case "COND_WAIT":
            startCondition();
            break;

        case "BLOCK_REST":
            goCondWait();
            break;

        case "TRIAL": {
            var captured = getCapturedTarget(mousePos);
            if (captured !== clickTarget) {
                trialErrorClicks++;
                updateTargetsFill(captured, clickTarget);
            } else {
                onTrialSuccess(mousePos);
            }
            break;
        }

        case "DONE":
            break;
    }
}

// =============================================================================
// ANIMATION
// =============================================================================

function ensureAnimation() {
    if (currentMovement !== "STATIC" && animFrame === null) {
        lastFrameTime = null;
        animFrame     = requestAnimationFrame(animate);
    }
}

function stopAnimation() {
    if (animFrame !== null) {
        cancelAnimationFrame(animFrame);
        animFrame = null;
    }
    lastFrameTime = null;
}

function animate(timestamp) {
    var dt = lastFrameTime !== null
        ? Math.min((timestamp - lastFrameTime) / 1000, 0.05)
        : 1 / 60;
    lastFrameTime = timestamp;

    if (targets.length && currentMovement !== "STATIC") {
        for (var i = 0; i < targets.length; i++) {
            var p = targets[i][0], v = targets[i][2], r = targets[i][1];
            if (currentMovement === "FAST") {
                advanceSplineTarget(targets[i], dt);
            } else {
                // SLOW: linear bounce
                p[0] += v[0] * dt; p[1] += v[1] * dt;
                var m = r + 6;
                if (p[0] < m)     { p[0] = m;     v[0] =  Math.abs(v[0]); }
                if (p[0] > W - m) { p[0] = W - m; v[0] = -Math.abs(v[0]); }
                if (p[1] < m)     { p[1] = m;     v[1] =  Math.abs(v[1]); }
                if (p[1] > H - m) { p[1] = H - m; v[1] = -Math.abs(v[1]); }
            }
        }
        if (svg) {
            svg.selectAll(".targetCircles")
                .attr("cx", function(d) { return d[0][0]; })
                .attr("cy", function(d) { return d[0][1]; });
        }
    }
    animFrame = requestAnimationFrame(animate);
}

// =============================================================================
// ENTRY POINT
// =============================================================================

function startSession() {
    participant    = parseInt(document.getElementById("participant").value, 10) || 1;
    var famVal     = parseInt(document.getElementById("famTrials").value, 10);
    famTrialsTotal = isNaN(famVal) || famVal < 0 ? FAM_TRIALS_DEFAULT : famVal;

    sessionData = {
        sessionStart:   new Date().toISOString(),
        sessionEnd:     null,
        participant:    participant,
        latinSquareRow: participant % 3,
        sequenceOrder:  LATIN_SQUARE[participant % 3].slice(),
        conditions:     [],
        sessionStats:   null
    };

    sessionSequence  = buildSessionSequence(participant);
    conditionIndex   = 0;
    famTrialsDone    = 0;
    trajectoryLog    = [];

    document.getElementById("setup").style.display = "none";
    d3.select("#saveNowBtn").style("display", "inline-block");
    initSVG();

    if (famTrialsTotal > 0) {
        goFamWait();
    } else {
        goCondWait();
    }
}