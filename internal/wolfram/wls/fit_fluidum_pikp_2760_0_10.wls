Print["FluiduM fit: ALICE 2.76 TeV Pb-Pb 0-10% (pi, K, p)"];

SetAttributes[runStep, HoldRest];
runStep[name_, expr_] := Module[{res},
  Print["[START] ", name];
  res = Check[ReleaseHold[Hold[expr]], $Failed];
  If[res === $Failed,
    Print["[FAIL] ", name];
    Print["Messages: ", $MessageList];
    Exit[2];
  ];
  Print["[OK] ", name];
  res
];

repoRoot = DirectoryName[DirectoryName[ExpandFileName[$ScriptCommandLine[[1]]]]];
fluidumPackageDir = FileNameJoin[{repoRoot, "..", "git", "pion-ratios", "material Klaus", "SlimFit", "Package"}];
fluidumPackageDir = ExpandFileName[fluidumPackageDir];
fluidumPackageFile = FileNameJoin[{fluidumPackageDir, "FluiduM.m"}];
If[!DirectoryQ[fluidumPackageDir] || !FileExistsQ[fluidumPackageFile],
  Print["ERROR: FluiduM package not found: ", fluidumPackageFile];
  Exit[1];
];

originalDir = Directory[];
fluidumProjectDir = DirectoryName[fluidumPackageDir];
SetDirectory[fluidumProjectDir];
runStep["Get[FluiduM.m]", Get[fluidumPackageFile]];

(* --- ALICE data (same slicing as your Mathematica notebook) --- *)
dataDir = FileNameJoin[{repoRoot, "data", "HEPData-ins1377750-v1-csv"}];
t1 = Import[FileNameJoin[{dataDir, "Table1.csv"}], "CSV"];
t2 = Import[FileNameJoin[{dataDir, "Table2.csv"}], "CSV"];
t3 = Import[FileNameJoin[{dataDir, "Table3.csv"}], "CSV"];

toNum[x_] := Quiet@Check[ToExpression[x], Indeterminate];
rowsNum[tab_, r1_, r2_] := Module[{block, rows, vals},
  block = Quiet@Check[tab[[r1 ;; r2]], {}];
  rows = Select[block, ListQ[#] && Length[#] >= 5 &];
  vals = Map[{toNum[#[[1]]], toNum[#[[4]]], Abs[toNum[#[[5]]]]} &, rows];
  Select[vals, VectorQ[#, NumericQ] && FreeQ[#, _ComplexInfinity | Indeterminate | Infinity | -Infinity] &]
];
combine010[a_, b_] := Module[{n, pt, y, e},
  If[!MatrixQ[a, NumericQ] || !MatrixQ[b, NumericQ],
    Print["ERROR: combine010 got non-numeric input. dims(a)=", Dimensions[a], " dims(b)=", Dimensions[b]];
    Exit[3];
  ];
  n = Min[Length[a], Length[b]];
  If[n <= 0,
    Print["ERROR: combine010 got empty inputs."];
    Exit[3];
  ];
  pt = a[[1 ;; n, 1]];
  y = (a[[1 ;; n, 2]] + b[[1 ;; n, 2]])/2.0;
  e = Sqrt[a[[1 ;; n, 3]]^2 + b[[1 ;; n, 3]]^2]/2.0;
  Transpose[{pt, y, e}]
];
invToDNdpt[data_] := Module[{pt = data[[All, 1]], y = data[[All, 2]], e = data[[All, 3]]},
  Transpose[{pt, 2.0*Pi*pt*y, 2.0*Pi*pt*e}]
];

pi005 = rowsNum[t1, 15, 77];
pi510 = rowsNum[t1, 84, 146];
k005 = rowsNum[t2, 15, 72];
k510 = rowsNum[t2, 79, 136];
p005 = rowsNum[t3, 15, 63];
p510 = rowsNum[t3, 70, 118];

Print["Data rows: pi005=", Length[pi005], " pi510=", Length[pi510], " k005=", Length[k005], " k510=", Length[k510], " p005=", Length[p005], " p510=", Length[p510]];

pi010 = invToDNdpt @ combine010[pi005, pi510];
k010 = invToDNdpt @ combine010[k005, k510];
p010 = invToDNdpt @ combine010[p005, p510];

(* fit windows *)
inWin[data_, lo_, hi_] := Select[data, lo <= #[[1]] <= hi &];
piFit = inWin[pi010, 0.5, 1.0];
kFit = inWin[k010, 0.0, 1.5];
pFit = inWin[p010, 0.0, 2.0];

(* --- FluiduM setup --- *)
alpha = 0.1;
rmax = 50;
npoints = 256;
myGrid = discretizationPoints[alpha, rmax, npoints];

derivMatrixFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsDerivativeMatrixA0.m"}];
sourceTermFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsSourceTermS0.m"}];
backgroundMatrices = runStep["getBackgroundMatrices", getBackgroundMatrices[derivMatrixFile, sourceTermFile]];

profilePath = "/Package/InitialConditions/LHCPbPb2760GeV/WeightFunction_1.txt";
centralities = {{0, 10}};

pTRange = {0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625,
  0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.05, 1.15, 1.25,
  1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35,
  2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1};

particleDataPCE = setParticleDataPCE2[{22, 111, 211, 221, 223, 311, 321, 2212}, pTRange];

TauOverKinEta = 5;
TauOverKinEtaIncrease = 3000;
etaa = 0.06125; etab = 0.02; etad = -0.70852; etadelta = 6.0; Tcrit = 0.156;
KinematicEta22[scale_] := Function[X, scale*(4/3)*(etaa*((X/Tcrit) - etad)^2 + etab/((X/Tcrit)^etadelta))];
TauShear22[Tchem_] := Function[X, If[X < Tchem, TauOverKinEta + (Tchem - X)*TauOverKinEtaIncrease, TauOverKinEta]];

fluidPropsFor[etaScale_, bulkAmp_] := Module[{FLP, FLshear, FLbulk},
  FLP = setThermodynamicProperties[];
  FLshear = setTransportPropertiesShear[FLP, "ShearViscosityOverEntropy" -> KinematicEta22[etaScale], "TauOverKinematicShearViscosity" -> TauShear22[0.145]];
  FLbulk = setTransportPropertiesBulk[FLP, "BulkViscosityOverEntropyAmplitude" -> bulkAmp, "BulkViscosityOverEntropyTmax" -> 0.175];
  Join[FLP, FLshear, FLbulk]
];

modelCache = <||>;
modelSpectra[norm_?NumericQ, tau0_?NumericQ, tkin_?NumericQ, tchem_?NumericQ] := Module[{key, fluidProperties, ic, sol, fo, spec},
  key = ToString@NumberForm[{norm, tau0, tkin, tchem}, {Infinity, 5}];
  If[KeyExistsQ[modelCache, key], Return[modelCache[key]]];

  fluidProperties = fluidPropsFor[2.1550, 0.0204];
  ic = setInitialConditionsCentralityAllFieldsNoNorm[
    centralities, norm, tau0, fluidProperties, myGrid,
    "InitialConditionsAll" -> False, "profilePath" -> profilePath
  ];
  sol = evolveBackground[40.0, fluidProperties, myGrid, ic[[1]], tkin, backgroundMatrices, "showProgressIndicator" -> False];
  fo = freezeOutBackground[sol, myGrid];
  spec = spectraExternalKernelBackground4D[particleDataPCE, fo, fluidProperties, tchem];
  (* spec dimensions: {8, npt} for one centrality in this call shape *)
  modelCache[key] = <|
    "pt" -> pTRange,
    "pi" -> spec[[3]],
    "k" -> spec[[7]],
    "p" -> spec[[8]]
  |>;
  modelCache[key]
];

bestScale[data_, modelVals_] := Module[{y = data[[All, 2]], e = data[[All, 3]], w, num, den},
  w = 1.0/(e^2);
  num = Total[y*modelVals*w];
  den = Total[modelVals^2*w];
  If[den <= 0, 1.0, num/den]
];

chi2For[norm_?NumericQ, tau0_?NumericQ, tkin_?NumericQ, tchem_?NumericQ] := Module[
  {m, mPiDNdpt, mKDNdpt, mPDNdpt, mpi, mk, mp, si, sk, sp, rpi, rk, rp, chi2},
  m = Quiet@Check[modelSpectra[norm, tau0, tkin, tchem], $Failed];
  If[m === $Failed, Return[10.^12]];
  (* FluiduM spectra are invariant-like: dN/(2 pi pT dpT dy). Convert to dN/dpTdy for comparison to ALICE dN/dpTdy. *)
  mPiDNdpt = 2.0*Pi*m["pt"]*m["pi"];
  mKDNdpt = 2.0*Pi*m["pt"]*m["k"];
  mPDNdpt = 2.0*Pi*m["pt"]*m["p"];
  mpi = Interpolation[Transpose[{m["pt"], mPiDNdpt}], InterpolationOrder -> 1] /@ piFit[[All, 1]];
  mk = Interpolation[Transpose[{m["pt"], mKDNdpt}], InterpolationOrder -> 1] /@ kFit[[All, 1]];
  mp = Interpolation[Transpose[{m["pt"], mPDNdpt}], InterpolationOrder -> 1] /@ pFit[[All, 1]];
  si = bestScale[piFit, mpi]; sk = bestScale[kFit, mk]; sp = bestScale[pFit, mp];
  rpi = (piFit[[All, 2]] - si*mpi)/piFit[[All, 3]];
  rk = (kFit[[All, 2]] - sk*mk)/kFit[[All, 3]];
  rp = (pFit[[All, 2]] - sp*mp)/pFit[[All, 3]];
  chi2 = Total[rpi^2] + Total[rk^2] + Total[rp^2];
  If[!NumericQ[chi2] || !FreeQ[chi2, _ComplexInfinity | Indeterminate | Infinity | -Infinity], Return[10.^12]];
  Print["chi2=", NumberForm[chi2, {8, 2}], "  norm=", norm, " tau0=", tau0, " Tkin=", tkin, " Tchem=", tchem];
  chi2
];

Print["Start fit (NelderMead)..."];
fit = NMinimize[
  {
    chi2For[norm, tau0, tkin, tchem],
    20 < norm < 60 && 0.4 < tau0 < 1.3 && 0.09 < tkin < 0.16 && 0.13 < tchem < 0.17
  },
  {norm, tau0, tkin, tchem},
  Method -> {"NelderMead"},
  MaxIterations -> 80
];

Print["Fit result: ", fit];

SetDirectory[originalDir];
