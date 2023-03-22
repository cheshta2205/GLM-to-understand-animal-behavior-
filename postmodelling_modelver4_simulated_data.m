%after running X:\Lab\dms_lesion\code\fit_behav_model_v9_updated.py

%for dms lesion 1 - j5_joy
model_folder = 'Z:\Lab\dms_lesion\antibiasing_model\model_output1';
data_folder = 'Z:\Lab\dms_lesion\antibiasing_model\data';

% load the model weights
weightleft = readtable([model_folder,filesep,'lefttapweights.csv']);
weightlefttap = weightleft(2:end,2:end);
weightlefttap = weightlefttap.Variables;

weightright = readtable([model_folder,filesep,'righttapweights.csv']);
weightrighttap = weightright(2:end,2:end);
weightrighttap = weightrighttap.Variables;

weightcenter = readtable([model_folder,filesep,'centtapweights.csv']);
weightcentertap = weightcenter(2:end,2:end);
weightcentertap = weightcentertap.Variables;

score= readtable([model_folder,filesep,'score_full.csv']);
score = score.Variables; score = score(2:end,2);

% load the current cue only score
cue_score = readtable([model_folder,filesep,'score_cue.csv']);
cue_score = cue_score.Variables; cue_score = cue_score(2:end,2);

% load the score std 
std_score = readtable([model_folder,filesep,'score_std.csv']);
std_score = std_score.Variables; std_score = std_score(2:end,2);
std_score = std_score./sqrt(numel(std_score));

% load the score std 
std_cuescore = readtable([model_folder,filesep,'score_cue_std.csv']);
std_cuescore = std_cuescore.Variables; std_cuescore = std_cuescore(2:end,2);
std_cuescore = std_cuescore./sqrt(numel(std_cuescore));

% load the score std 
actscorestd = readtable([model_folder,filesep,'score_actstd.csv']);
actscorestd  = actscorestd .Variables; actscorestd  = actscorestd (2:end,2);
actscorestd = actscorestd./sqrt(numel(actscorestd));

% load the previous action only score
act_score = readtable([model_folder,filesep,'score_act.csv']);
act_score = act_score.Variables; act_score = act_score(2:end,2);

% load the null score (only bias)
null_score= readtable([model_folder,filesep,'score_null.csv']);
null_score = null_score.Variables; null_score = null_score(2:end,2);

factors = {'CueL','CueC','CueR','Rew_L_1','Rew_C_1','Rew_R_1','nonRew_L_1','nonRew_C_1','nonRew_R_1','bias'};
factor_count = length(factors);

%% 

num_sessions = size(weightrighttap,1);
width = 10;

% model score and rat accuracy (basically plotting scorres to see which
% model can predict behavior best)
xaxx = 1:1:length(score);
figure(1)
h1=plot(xaxx,smoothdata(score,'gauss',width),'-r','LineWidth',8); 
% shadedErrorBar(xaxx,smoothdata(score,'gauss',width),std_score);
hold on
h2=plot(xaxx,smoothdata(cue_score,'gauss',width),'-g','LineWidth',8);
% shadedErrorBar(xaxx,smoothdata(cue_score,'gauss',width),std_cuescore);
hold on
h3=plot(xaxx,smoothdata(act_score,'gauss',width),'-b','LineWidth',8); 
% shadedErrorBar(xaxx,smoothdata(act_score,'gauss',width),actscorestd);
h4=plot(xaxx,smoothdata(null_score,'gauss',width),'-k','LineWidth',8); 

% shadedErrorBar(xaxx,smoothdata(null_score,'gauss',width),actscorestd);
legend([h1,h2,h3,h4],'full model','cue Model','previous actions','null score','LineWidth',5,'FontSize',20);

ylabel('Model prediction scores');
xlabel('Block of 100 trials');
hold on 
ylim([0 1]);
hold on

h4 = gca;
h4.XAxis.LineWidth = 8;
h4.YAxis.LineWidth = 8;
h4.XAxis.FontSize = 20;
h4.YAxis.FontSize = 20;
hold off 

c = distinguishable_colors(25);

figure(2)
for k = 1:size(weightrighttap,2)
    plot(smoothdata(weightrighttap(:,k),'gauss',5),'MarkerSize',10,'LineWidth',2,'Color',c(k,:));
    hold on 
end
hold on
legend(factors)
ylabel('Weights')
xlabel('Session')
title('right tap')
hold on
box off
h2 = gca 
h2.XAxis.LineWidth = 5;
h2.YAxis.LineWidth = 5;
h2.XAxis.FontSize = 15;
h2.YAxis.FontSize = 15;
hold off 

figure(3)
for k = 1:size(weightlefttap,2)
    plot(smoothdata(weightlefttap(:,k),'gauss',5),'MarkerSize',10,'LineWidth',2,'Color',c(k,:));
    hold on 
end
hold on
legend(factors)
ylabel('Weights')
xlabel('Session')
title('Left tap')
hold on
box off
h2 = gca 
h2.XAxis.LineWidth = 5;
h2.YAxis.LineWidth = 5;
h2.XAxis.FontSize = 15;
h2.YAxis.FontSize = 15;
hold off 

figure(4)
for k = 1:size(weightcentertap,2)
    plot(smoothdata(weightcentertap(:,k),'gauss',5),'MarkerSize',10,'LineWidth',2,'Color',c(k,:));
    hold on 
end
hold on
legend(factors)
ylabel('Weights')
xlabel('Session')
title('Center tap')
hold on
box off
h2 = gca 
h2.XAxis.LineWidth = 5;
h2.YAxis.LineWidth = 5;
h2.XAxis.FontSize = 15;
h2.YAxis.FontSize = 15;
hold off 
