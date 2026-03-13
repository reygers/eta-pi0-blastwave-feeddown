Print["Loading FluiduM package..."];

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

(* In wolframscript, NotebookDirectory[] is not available (no front end).
   Use an explicit package directory, optionally overridable via CLI:
   wolframscript -file script.wls "/path/to/Package"
*)
fluidumPackageDir = If[
  Length[$ScriptCommandLine] >= 2,
  $ScriptCommandLine[[2]],
  "/Users/reygers/uni/my_papers/ideas/eta-to-pi0-ratio/git/pion-ratios/material Klaus/SlimFit/Package"
];

If[!DirectoryQ[fluidumPackageDir],
  Print["ERROR: FluiduM package directory not found: ", fluidumPackageDir];
  Exit[1];
];

If[FreeQ[$Path, fluidumPackageDir], AppendTo[$Path, fluidumPackageDir]];
fluidumPackageFile = If[
  Length[$ScriptCommandLine] >= 3,
  $ScriptCommandLine[[3]],
  FileNameJoin[{fluidumPackageDir, "FluiduM.m"}]
];
If[!FileExistsQ[fluidumPackageFile],
  Print["ERROR: Could not find FluiduM package file in: ", fluidumPackageDir];
  Exit[1];
];

runStep["Get[FluiduM package]", Get[fluidumPackageFile]];

Print["Loaded FluiduM` from: ", fluidumPackageFile];
originalDir = Directory[];
(* FluiduM internals build paths as StringJoin[Directory[], "/Package/..."].
   Therefore working directory must be the parent of "Package", not "Package" itself. *)
fluidumProjectDir = DirectoryName[fluidumPackageDir];
SetDirectory[fluidumProjectDir];
Print["Working directory set to: ", Directory[]];

(* Here we set up the grid and load the matrices *)
alpha = 0.1;
rmax = 50; (* maximum of radial grid *)
npoints = 2*128; (* number of points on radial grid *)
tfinal = 30;
myGrid = discretizationPoints[alpha, rmax, npoints];
grid = myGrid[["DiscretizationGrid"]];(* just the radial grid *)

derivMatrixFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsDerivativeMatrixA0.m"}];
sourceTermFile = FileNameJoin[{fluidumPackageDir, "EquationOfMotion", "FluidEquationsSourceTermS0.m"}];

If[!FileExistsQ[derivMatrixFile],
  Print["ERROR: Missing file: ", derivMatrixFile];
  Exit[1];
];
If[!FileExistsQ[sourceTermFile],
  Print["ERROR: Missing file: ", sourceTermFile];
  Exit[1];
];

backgroundMatrices = runStep[
  "getBackgroundMatrices",
  getBackgroundMatrices[derivMatrixFile, sourceTermFile]
];

If[!DirectoryQ["plots"], CreateDirectory["plots", CreateIntermediateDirectories -> True]];

(* Fluid properties *)
Tc = 0.156;
\[Eta]s = 0.2;
\[Eta][T_] := 0.2*4/3*(0.0613*(T/0.156 + 0.709)^2 + 0.02/(T/0.156)^6)

TauOverKinEta = 5;
TauOverKinEtaIncrease = 3000;

(* ShearViscosityOverEntropy parametrisation Jan Pawlowski *)
etaa = 0.06125;
etab = 0.02;
etad = -0.70852;
etadelta = 6.0;
Tcrit = 0.156;

KinematicEta22[scale_] := Function[X,
     scale * (4/3) * ( 
     etaa * ( (X / Tcrit) - etad )^2 + etab / ((X / Tcrit)^etadelta) )
   ];

TauShear22[Tchem_] := Function[X,
     If[X < Tchem,
        TauOverKinEta + (Tchem - X)*TauOverKinEtaIncrease,
        TauOverKinEta
      ]
   ];

tfinal = 40;
norm = 35.8863;
eta = 2.1550;
bulk = 0.0204;
tau0 = 1.1112;
TKin = 0.1236;
TChem = 0.1457;

etaa = 0.06125;
etab = 0.02;
etad = -0.70852;
etadelta = 6.0;
Tcrit = 0.156;

KinematicEta22[scale_] := 
  Function[X, 
   scale*(4/3)*(etaa*((X/Tcrit) - etad)^2 + etab/((X/Tcrit)^etadelta))];

TauShear22[Tchem_] := 
  Function[X, 
   If[X < Tchem, TauOverKinEta + (Tchem - X)*TauOverKinEtaIncrease, 
    TauOverKinEta]];

FLP = setThermodynamicProperties[];

listShearTchem = {{2.155`, 0.1457`}};

listBulk = {0.0204`};

FLshear = <|
   Table[shearTchem[[1]] -> 
     setTransportPropertiesShear[FLP, 
      "ShearViscosityOverEntropy" -> KinematicEta22[shearTchem[[1]]], 
      "TauOverKinematicShearViscosity" -> 
       TauShear22[shearTchem[[2]]]], {shearTchem, listShearTchem}]|>;

FLbulk = <|
   Table[bulk -> 
     setTransportPropertiesBulk[FLP, 
      "BulkViscosityOverEntropyAmplitude" -> bulk, 
      "BulkViscosityOverEntropyTmax" -> 0.175], {bulk, listBulk}]|>;

fluidProperties = Join[FLP, FLshear[eta], FLbulk[bulk]];


(* Define centrality bins *)
centralities = {{0, 10}, {20, 50}};

(* FluiduM parameters *)
norm276 = 36.0;
tau0276 = 0.76;
Tkin = 0.122;
Tchem = 0.141;

profilePath276Abs = FileNameJoin[{fluidumPackageDir, "InitialConditions", "LHCPbPb2760GeV", "WeightFunction_1.txt"}];
If[!FileExistsQ[profilePath276Abs],
  Print["ERROR: Missing profile file: ", profilePath276Abs];
  Exit[1];
];
(* FluiduM code does Import[StringJoin[Directory[], profilePath]], with Directory[] = fluidumProjectDir. *)
profilePath276 = "/Package/InitialConditions/LHCPbPb2760GeV/WeightFunction_1.txt";

IC276 = setInitialConditionsCentralityAllFieldsNoNorm[centralities, 
   norm276, tau0276, fluidProperties, myGrid, 
   "InitialConditionsAll" -> False, 
   "profilePath" -> 
    "/Package/InitialConditions/LHCPbPb2760GeV/WeightFunction_1.txt"];

(* Evolve background *)
solBG276 = 
  evolveBackground[tfinal, fluidProperties, myGrid, #, Tkin, 
     backgroundMatrices] & /@ IC276;

foBG = freezeOutBackground[#, myGrid] & /@ solBG276;

pTRange = {0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, \
0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.05, 1.15, 1.25, \
1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, \
2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1}

particleData = 
  setParticleDataPCE2[{22, 111, 211, 221, 223, 311, 321, 2212}, 
   pTRange]; 

particleDataPCE = 
  setParticleDataPCE[{22, 111, 211, 221, 223, 311, 321, 2212}, 
   pTRange];

spec276PCE = runStep[
  "spectraExternalKernelBackground4D /@ foBG",
  (spectraExternalKernelBackground4D[particleDataPCE, #, fluidProperties, Tchem] &) /@ foBG
];

Print["Dimensions[spec276PCE] = ", Dimensions[spec276PCE]];

EtaToPi0Ratio = spec276PCE[[All, 4]]/spec276PCE[[All, 2]];
KaonToPionRatio = spec276PCE[[All, 7]]/spec276PCE[[All, 3]];
ProtonToPionRatio = spec276PCE[[All, 8]]/spec276PCE[[All, 3]];
KaonToEtaRatio = spec276PCE[[All, 7]]/spec276PCE[[All, 4]];
PionToPi0Ratio = spec276PCE[[All, 3]]/spec276PCE[[All, 2]];
doubleRatio = EtaToPi0Ratio/KaonToPionRatio;

(* Plot ratios *)
makeRatioPlot[y_, label_] := Module[{yy = y, curves, labs},
  curves = If[ArrayDepth[yy] == 1, {yy}, yy];
  labs = Table[
    ToString[centralities[[i, 1]]] <> "–" <> ToString[centralities[[i, 2]]] <> "%",
    {i, 1, Min[Length[curves], Length[centralities]]}
  ];
  ListLinePlot[
    Table[Transpose[{pTRange, curves[[i]]}], {i, 1, Length[curves]}],
    Frame -> True,
    FrameLabel -> {"pT (GeV/c)", label},
    PlotRange -> All,
    PlotLegends -> labs
  ]
];

p1 = makeRatioPlot[EtaToPi0Ratio, "eta/pi0"];
p2 = makeRatioPlot[KaonToPionRatio, "K+/pi+"];
p3 = makeRatioPlot[ProtonToPionRatio, "p/pi+"];
p4 = makeRatioPlot[KaonToEtaRatio, "K+/eta"];
p5 = makeRatioPlot[PionToPi0Ratio, "pi+/pi0"];
p6 = makeRatioPlot[doubleRatio, "(eta/pi0)/(K/pi)"];

Export["plots/eta_to_pi0_ratio.pdf", p1];
Export["plots/kaon_to_pion_ratio.pdf", p2];
Export["plots/proton_to_pion_ratio.pdf", p3];
Export["plots/kaon_to_eta_ratio.pdf", p4];
Export["plots/pion_to_pi0_ratio.pdf", p5];
Export["plots/double_ratio.pdf", p6];

SetDirectory[originalDir];
