clear all ; clc; close all;

warning('off','all');


%% Apply PCA
%Load and read data from a csv file to a .mat file
CleanedData = table2array(readtable('Data.xlsx'));
CleanedData = CleanedData(1:14733 , :);
Features = CleanedData(: , 1);
Features2 = CleanedData(: , 3:end);
Features = horzcat(Features , Features2);
Labels = CleanedData(: , 2);

% Apply PCA
[coeff,score,latent,~,explained,mu] = pca(Features);

reducedDimension = score(:,1:3);

% h = histogram(explained)
%% Apply NCA
mdl = fsrnca(Features,Labels,'Verbose',1,'Lambda',0.5/14733);
figure()
plot(mdl.FeatureWeights,'ro')
grid on
xlabel('Feature index')
ylabel('Feature weight')

%% Normalization
X=reducedDimension;
XNorm = X;
mu = zeros(1, size(X, 2));
stddev = zeros(1, size(X, 2));

% Calculates mean and std dev for each feature
for i=1:size(mu,2)
    mu(1,i) = mean(X(:,i));
    stddev(1,i) = std(X(:,i));
    XNorm(:,i) = (X(:,i)-mu(1,i))/stddev(1,i);
end

reducedDimension=XNorm;

%% Training
% Fitting problem with a Neural Network
x = reducedDimension';
t = Labels';
trainFcn = 'trainscg';
hiddenLayerSize = 1;
net = fitnet(hiddenLayerSize,trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
[net,tr] = train(net,x,t);
y = net(x);

NNact = t(tr.testInd);
NNpred = y(tr.testInd);
trainTargets = t .* tr.trainMask{1};
testTargets = t .* tr.testMask{1};


TrFeatures = x(:,tr.trainInd)';
TrLabels = t(tr.trainInd)';
% Fit linear regression model
MdlLinearReg = fitlm(TrFeatures,TrLabels,'linear');
% Fit linear regression model to high-dimensional data
MdlLinearSGD = fitrlinear(TrFeatures,TrLabels,'Learner','leastsquares');
% Fit linear regression model using SVM
MdlSVM = fitrsvm(TrFeatures,TrLabels);
% Fit a Gaussian process regression (GPR) model
MdlGP = fitrgp(TrFeatures,TrLabels,'BasisFunction', 'constant','KernelFunction', 'exponential','Standardize', true);
% Fit ensemble of learners for regression
template = templateTree('MinLeafSize', 8);
MdlEnsemble = fitrensemble(TrFeatures,TrLabels,'Method', 'LSBoost','NumLearningCycles', 30,'Learners', template,'LearnRate', 0.1);
% Fit binary decision tree for regression
MdlTree = fitrtree(TrFeatures,TrLabels,'MinLeafSize', 4,'Surrogate', 'off');
% Perform stepwise regression
MdlStepWise = stepwiselm(TrFeatures,TrLabels,'linear','Upper', 'interactions','NSteps', 1000,'Verbose', 0);





% Training Predictions

TrpredLinearReg = predict(MdlLinearReg,TrFeatures);
TrpredLinearSGD = predict(MdlLinearSGD,TrFeatures);
TrpredSVM       = predict(MdlSVM,TrFeatures);
TrpredGP        = predict(MdlGP,TrFeatures);
TrpredEnsemble  = predict(MdlEnsemble,TrFeatures);
TrpredTree      = predict(MdlTree,TrFeatures);
TrpredStepWise  = predict(MdlStepWise,TrFeatures);


% Training Performance
% Mean Absolute Error

TrMAELinearReg  = mean(abs(TrLabels-TrpredLinearReg));
TrMAELinearSGD  = mean(abs(TrLabels-TrpredLinearSGD));
TrMAESVM        = mean(abs(TrLabels-TrpredSVM));
TrMAEGP         = mean(abs(TrLabels-TrpredGP));
TrMAEEnsemble   = mean(abs(TrLabels-TrpredEnsemble));
TrMAETree       = mean(abs(TrLabels-TrpredTree));
TrMAEStepWise   = mean(abs(TrLabels-TrpredStepWise));
TrMAENN         = mae(net,trainTargets,y);
% % Mean Squared Error
% 
TrMSELinearReg = mean((TrLabels - TrpredLinearReg).^2);
TrMSELinearSGD = mean((TrLabels - TrpredLinearSGD).^2);
TrMSESVM       = mean((TrLabels - TrpredSVM).^2);
TrMSEGP        = mean((TrLabels - TrpredGP).^2);
TrMSEEnsemble  = mean((TrLabels - TrpredEnsemble).^2);
TrMSETree      = mean((TrLabels - TrpredTree).^2);
TrMSEStepWise  = mean((TrLabels - TrpredStepWise).^2);
TrMSENN        = mse(net,trainTargets,y);
% % Root Mean Squared Error
% 
TrRMSELinearReg = sqrt(mean((TrLabels - TrpredLinearReg).^2));
TrRMSELinearSGD = sqrt(mean((TrLabels - TrpredLinearSGD).^2));
TrRMSESVM       = sqrt(mean((TrLabels - TrpredSVM).^2));
TrRMSEGP        = sqrt(mean((TrLabels - TrpredGP).^2));
TrRMSEEnsemble  = sqrt(mean((TrLabels - TrpredEnsemble).^2));
TrRMSETree      = sqrt(mean((TrLabels - TrpredTree).^2));
TrRMSEStepWise  = sqrt(mean((TrLabels - TrpredStepWise).^2));
TrRMSENN        = sqrt(TrMSENN);

%% Testing
TesFeatures = x(:,tr.testInd)';
TesLabels = t(tr.testInd)';

TespredLinearReg  = predict(MdlLinearReg,TesFeatures);
TespredLinearSGD  = predict(MdlLinearSGD,TesFeatures);
TespredSVM        = predict(MdlSVM,TesFeatures);
TespredGP         = predict(MdlGP,TesFeatures);
TespredEnsemble   = predict(MdlEnsemble,TesFeatures);
TespredTree       = predict(MdlTree,TesFeatures);
TespredStepWise   = predict(MdlStepWise,TesFeatures);

% Testing Performance
% Mean Absolute Value

TesMAELinearReg  = mean(abs(TesLabels-TespredLinearReg));
TesMAELinearSGD  = mean(abs(TesLabels-TespredLinearSGD));
TesMAESVM        = mean(abs(TesLabels-TespredSVM));
TesMAEGP         = mean(abs(TesLabels-TespredGP));
TesMAEEnsemble   = mean(abs(TesLabels-TespredEnsemble));
TesMAETree       = mean(abs(TesLabels-TespredTree));
TesMAEStepWise   = mean(abs(TesLabels-TespredStepWise));
TesMAENN         = mae(net,testTargets,y);
% Mean Squared Error

TesMSELinearReg = mean((TesLabels - TespredLinearReg).^2);
TesMSELinearSGD = mean((TesLabels - TespredLinearSGD).^2);
TesMSESVM       = mean((TesLabels - TespredSVM).^2);
TesMSEGP        = mean((TesLabels - TespredGP).^2);
TesMSEEnsemble  = mean((TesLabels - TespredEnsemble).^2);
TesMSETree      = mean((TesLabels - TespredTree).^2);
TesMSEStepWise  = mean((TesLabels - TespredStepWise).^2);
TesMSENN        = mse(net,testTargets,y);
% Root Mean Squared Error

TesRMSELinearReg = sqrt(mean((TesLabels - TespredLinearReg).^2));
TesRMSELinearSGD = sqrt(mean((TesLabels - TespredLinearSGD).^2));
TesRMSESVM       = sqrt(mean((TesLabels - TespredSVM).^2));
TesRMSEGP        = sqrt(mean((TesLabels - TespredGP).^2));
TesRMSEEnsemble  = sqrt(mean((TesLabels - TespredEnsemble).^2));
TesRMSETree      = sqrt(mean((TesLabels - TespredTree).^2));
TesRMSEStepWise  = sqrt(mean((TesLabels - TespredStepWise).^2));
TesRMSENN        = sqrt(TesMSENN);



%% Accuracy : below 70 is referred to as hypoglycemia, while over 130 mg/dL is called hyperglycemia.
LActual        = TransformLabels(TesLabels);
LPredLinearReg = TransformLabels(TespredLinearReg);
LPredLinearSGD = TransformLabels(TespredLinearSGD);
LPredSVM       = TransformLabels(TespredSVM);
LPredGP        = TransformLabels(TespredGP);
LPredEnsemble  = TransformLabels(TespredEnsemble);
LPredTree      = TransformLabels(TespredTree);
LPredStepWise  = TransformLabels(TespredStepWise);
LPredNN        = TransformLabels(NNpred);



AccuLinearReg = findaccuracy(LPredLinearReg,LActual);
AccuLinearSGD = findaccuracy(LPredLinearSGD,LActual);
AccuSVM       = findaccuracy(LPredSVM,LActual);
AccuGP        = findaccuracy(LPredGP,LActual);
AccuEnsemble  = findaccuracy(LPredEnsemble,LActual);
AccuTree      = findaccuracy(LPredTree,LActual);
AccuStepWise  = findaccuracy(LPredStepWise,LActual);
AccuNN        = findaccuracy(LPredNN,LActual);

AccuLinearReg = round(AccuLinearReg,4);
AccuLinearSGD = round(AccuLinearSGD,4);
AccuSVM       = round(AccuSVM,4);
AccuGP        = round(AccuGP,4);
AccuEnsemble  = round(AccuEnsemble,4);
AccuTree      = round(AccuTree,4);
AccuStepWise  = round(AccuStepWise,4);
AccuNN        = round(AccuNN,4);

cm = confusionchart(LActual,LPredLinearReg);

fig = tiledlayout(2,4,'TileSpacing','Compact');

% Tile 1
nexttile
confusionchart(LActual,LPredLinearReg);
title(['LR (',num2str(AccuLinearReg*100),'%)'])

% Tile 2
nexttile
confusionchart(LActual,LPredLinearSGD);
title(['LRSGD (',num2str(AccuLinearSGD*100),'%)'])

% Tile 3
nexttile
confusionchart(LActual,LPredSVM);
title(['SVM (',num2str(AccuSVM*100),'%)'])

% Tile 4
nexttile
confusionchart(LActual,LPredGP);
title(['GPR (',num2str(AccuGP*100),'%)'])


% Tile 5
nexttile
confusionchart(LActual,LPredEnsemble);
title(['BSTE (',num2str(AccuEnsemble*100),'%)'])

% Tile 6
nexttile
confusionchart(LActual,LPredTree);
title(['BDT (',num2str(AccuTree*100),'%)'])

% Tile 7
nexttile
confusionchart(LActual,LPredStepWise);
title(['SW (',num2str(AccuStepWise*100),'%)'])

% Tile 8
nexttile
confusionchart(LActual,LPredNN);
title(['ANN (',num2str(AccuNN*100),'%)'])


%%

Names = {'Mean Absolute Error';'Mean Square Error';'Root Mean Squarred Error'};

TrLinearReg = [TrMAELinearReg;TrMSELinearReg;TrRMSELinearReg];
TrLinearSGD = [TrMAELinearSGD;TrMSELinearSGD;TrRMSELinearSGD];
TrSVM       = [TrMAESVM;TrMSESVM;TrRMSESVM];
TrGP        = [TrMAEGP;TrMSEGP;TrRMSEGP];
TrEnsemble  = [TrMAEEnsemble;TrMSEEnsemble;TrRMSEEnsemble];
TrTree      = [TrMAETree;TrMSETree;TrRMSETree];
TrStepWise  = [TrMAEStepWise;TrMSEStepWise;TrRMSEStepWise];
TrNN        = [TrMAENN;TrMSENN;TrRMSENN];

TrainingResults = table(Names,TrLinearReg,TrLinearSGD,TrSVM,TrGP,TrEnsemble,TrTree,TrStepWise,TrNN);
% 

TesLinearReg = [TesMAELinearReg;TesMSELinearReg;TesRMSELinearReg];
TesLinearSGD = [TesMAELinearSGD;TesMSELinearSGD;TesRMSELinearSGD];
TesSVM       = [TesMAESVM;TesMSESVM;TesRMSESVM];
TesGP        = [TesMAEGP;TesMSEGP;TesRMSEGP];
TesEnsemble  = [TesMAEEnsemble;TesMSEEnsemble;TesRMSEEnsemble];
TesTree      = [TesMAETree;TesMSETree;TesRMSETree];
TesStepWise  = [TesMAEStepWise;TesMSEStepWise;TesRMSEStepWise];
TesNN        = [TesMAENN;TesMSENN;TesRMSENN];

TestingResults = table(Names,TesLinearReg,TesLinearSGD,TesSVM,TesGP,TesEnsemble,TesTree,TesStepWise,TesNN);



%% Visualization

fig = tiledlayout(2,4,'TileSpacing','Compact');

% Tile 1
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredLinearReg,'k','LineWidth',2);
legend('True response','Predicted values','Location','Best');
hold off
title('LR')

% Tile 2
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredLinearSGD,'k','LineWidth',2);
hold off
title('LRSGD')

% Tile 3
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredSVM,'k','LineWidth',2);
hold off
title('SVM')

% Tile 4
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredGP,'k','LineWidth',2);
hold off
title('GPR')


% Tile 5
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredEnsemble,'k','LineWidth',2);
hold off
title('BSTE')

% Tile 6
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredTree,'k','LineWidth',2);
hold off
title('BDT')

% Tile 7
nexttile
plot(TesLabels,'r','LineWidth',2);
hold on;
plot(TespredStepWise,'k','LineWidth',2);
hold off
title('SW')

% Tile 8
nexttile
plot(NNact,'r','LineWidth',2);
hold on;
plot(NNpred,'k','LineWidth',2);
hold off
title('ANN')


