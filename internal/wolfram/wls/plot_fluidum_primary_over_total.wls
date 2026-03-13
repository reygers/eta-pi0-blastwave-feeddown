Print["FluiduM primary/total ratios (pi+, pi0, eta, K+)"];

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
fluidumPackageDir = If[
  Length[$ScriptCommandLine] >= 2,
  $ScriptCommandLine[[2]],
  ExpandFileName @ FileNameJoin[{repoRoot, "..", "git", "pion-ratios", "material Klaus", "SlimFit", "Package"}]
];
fluidumPackageFile = If[
  Length[$ScriptCommandLine] >= 3,
  $ScriptCommandLine[[3]],
  FileNameJoin[{fluidumPackageDir, "FluiduM.m"}]
];
If[!DirectoryQ[fluidumPackageDir] || !FileExistsQ[fluidumPackageFile],
  Print["ERROR: FluiduM package not found: ", fluidumPackageFile];
  Exit[1];
];

scriptDir = DirectoryName[ExpandFileName[$ScriptCommandLine[[1]]]];
plotDir = FileNameJoin[{scriptDir, "plots"}];
If[!DirectoryQ[plotDir], CreateDirectory[plotDir, CreateIntermediateDirectories -> True]];

originalDir = Directory[];
fluidumProjectDir = DirectoryName[fluidumPackageDir];
SetDirectory[fluidumProjectDir];

runStep["Get[FluiduM package]", Get[fluidumPackageFile]];
Print["Loaded FluiduM from: ", fluidumPackageFile];
Print["Working directory: ", Directory[]];

(* Parameters from example script *)
alpha = 0.1;
rmax = 50;
npoints = 2*128;
tfinal = 40.0;
norm276 = 36.0;
tau0276 = 0.76;
Tkin = 0.122;
Tchem = 0.141;
eta = 2.1550;
bulk = 0.0204;
centralities = {{0, 10}, {20, 50}};

TauOverKinEta = 5;
TauOverKinEtaIncrease = 3000;
etaa = 0.06125; etab = 0.02; etad = -0.70852; etadelta = 6.0; Tcrit = 0.156;
KinematicEta22[scale_] := Function[X, scale*(4/3)*(etaa*((X/Tcrit) - etad)^2 + etab/((X/Tcrit)^etadelta))];
TauShear22[tchem_] := Function[X, If[X < tchem, TauOverKinEta + (tchem - X)*TauOverKinEtaIncrease, TauOverKinEta]];

myGrid = discretizationPoints[alpha, rmax, npoints];
derivMatrixFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsDerivativeMatrixA0.m"}];
sourceTermFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsSourceTermS0.m"}];
backgroundMatrices = runStep["getBackgroundMatrices", getBackgroundMatrices[derivMatrixFile, sourceTermFile]];

FLP = setThermodynamicProperties[];
FLshear = setTransportPropertiesShear[
  FLP,
  "ShearViscosityOverEntropy" -> KinematicEta22[eta],
  "TauOverKinematicShearViscosity" -> TauShear22[Tchem]
];
FLbulk = setTransportPropertiesBulk[
  FLP,
  "BulkViscosityOverEntropyAmplitude" -> bulk,
  "BulkViscosityOverEntropyTmax" -> 0.175
];
fluidProperties = Join[FLP, FLshear, FLbulk];

profilePath = "/Package/InitialConditions/LHCPbPb2760GeV/WeightFunction_1.txt";
IC276 = runStep[
  "setInitialConditionsCentralityAllFieldsNoNorm",
  setInitialConditionsCentralityAllFieldsNoNorm[
    centralities, norm276, tau0276, fluidProperties, myGrid,
    "InitialConditionsAll" -> False, "profilePath" -> profilePath
  ]
];

solBG276 = runStep[
  "evolveBackground /@ IC276",
  (evolveBackground[tfinal, fluidProperties, myGrid, #, Tkin, backgroundMatrices, "showProgressIndicator" -> False] &) /@ IC276
];
foBG = runStep["freezeOutBackground /@ solBG276", (freezeOutBackground[#, myGrid] &) /@ solBG276];

pTRange = {0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625,
  0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.05, 1.15, 1.25,
  1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35,
  2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1};

species = {
  <|"pid" -> 211, "key" -> "pi_plus", "label" -> "pi+"|>,
  <|"pid" -> 111, "key" -> "pi0", "label" -> "pi0"|>,
  <|"pid" -> 221, "key" -> "eta", "label" -> "eta"|>,
  <|"pid" -> 321, "key" -> "k_plus", "label" -> "K+"|>
};

massList = <|"211" -> 0.1396, "111" -> 0.1350, "221" -> 0.5470, "321" -> 0.4937|>;
degList = <|"211" -> 1, "111" -> 1, "221" -> 1, "321" -> 1|>;
statList = <|"211" -> 1, "111" -> 1, "221" -> 1, "321" -> 1|>;
nameList = <|"211" -> "pi0139plu", "111" -> "pi0135zer", "221" -> "eta0547zer", "321" -> "Ka0492plu"|>;

makeParticleData[kernelDir_] := Association @ Table[
  ToString[s["pid"]] -> <|
    "Mass" -> massList[ToString[s["pid"]]],
    "Degeneracy" -> degList[ToString[s["pid"]]],
    "StatVariable" -> statList[ToString[s["pid"]]],
    "NameParticle" -> nameList[ToString[s["pid"]]],
    "BackgroundKernel" -> Import[StringJoin[Directory[], kernelDir, ToString[s["pid"]], If[StringContainsQ[kernelDir, "kernel_full_PCE_total"], "_full_PCE_total.mx", "_full_PCE.mx"]]],
    "pTList" -> pTRange,
    "CentralityDependentpT" -> False
  |>,
  {s, species}
];

pdTotal = makeParticleData["/Package/Spectra_Kernel/Background_Kernels/kernel_full_PCE_total/"];
pdPrimary = makeParticleData["/Package/Spectra_Kernel/Background_Kernels/kernel_full_PCE/"];

specTotal = runStep[
  "spectraExternalKernelBackground4D (total) /@ foBG",
  (spectraExternalKernelBackground4D[pdTotal, #, fluidProperties, Tchem] &) /@ foBG
];
specPrimary = runStep[
  "spectraExternalKernelBackground4D (primary) /@ foBG",
  (spectraExternalKernelBackground4D[pdPrimary, #, fluidProperties, Tchem] &) /@ foBG
];

ratio = specPrimary/specTotal;

colors = {Blue, Red, Darker[Green], Purple};
labels = species[[All, "label"]];

Do[
  centLabel = ToString[centralities[[ic, 1]]] <> "–" <> ToString[centralities[[ic, 2]]] <> "%";
  curves = Table[Transpose[{pTRange, ratio[[ic, is, All]]}], {is, 1, Length[species]}];

  plt = ListLinePlot[
    curves,
    Frame -> True,
    FrameLabel -> {"pT (GeV/c)", "primary/total"},
    PlotRange -> {{0.2, 3.2}, {0.0, 1.05}},
    PlotStyle -> Table[{colors[[i]], Thick}, {i, 1, Length[species]}],
    PlotLegends -> labels,
    GridLines -> Automatic,
    ImageSize -> 800,
    PlotLabel -> ("FluiduM primary/total, Pb–Pb 2.76 TeV, " <> centLabel)
  ];

  outPdf = FileNameJoin[{plotDir, "fluidum_primary_over_total_" <> StringReplace[centLabel, {"–" -> "_", "%" -> "", " " -> ""}] <> ".pdf"}];
  Export[outPdf, plt];
  Print["Wrote ", outPdf];

  csvHeader = StringRiffle[Join[{"pt"}, species[[All, "key"]]], ","];
  csvData = Transpose @ Join[{pTRange}, Table[ratio[[ic, is, All]], {is, 1, Length[species]}]];
  outCsv = FileNameJoin[{plotDir, "fluidum_primary_over_total_" <> StringReplace[centLabel, {"–" -> "_", "%" -> "", " " -> ""}] <> ".csv"}];
  Export[outCsv, Prepend[csvData, StringSplit[csvHeader, ","]]];
  Print["Wrote ", outCsv];
,
  {ic, 1, Length[centralities]}
];

SetDirectory[originalDir];
Print["Done."];
