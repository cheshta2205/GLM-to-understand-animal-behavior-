%try for j5

files = dir('W:\Lab\dms_lesion\data_modelver4\csv_t5tiffany_control\*.csv');
for i = 1:numel(files)
filename{i} = files(i).name;
end
ii=i./2;
for iii = 1:ii
    filesip{iii} = ['inputtable_session_',num2str(iii)];
    loaded_cueinput{1,iii} = readtable(filesip{1,iii});
end

for iii = 1:ii
    filesop{iii} = ['outputtable_session_',num2str(iii)];
    loaded_actionoutput{1,iii} = readtable(filesop{1,iii});
end

for iii = 1:ii
actions{1,iii} = nan(height(loaded_actionoutput{1,iii}),1);
cue_ips{1,iii} = nan(height(loaded_cueinput{1,iii}),1);
end

for iii = 1:ii
id_left{1,iii} = find(loaded_actionoutput{1,iii}.tapL==1);
id_center{1,iii} = find(loaded_actionoutput{1,iii}.tapC==1);
id_right{1,iii} = find(loaded_actionoutput{1,iii}.tapR==1);
%left=1, center=2, right=3
actions{1,iii}(id_left{1,iii})=1;
actions{1,iii}(id_center{1,iii})=2;
actions{1,iii}(id_right{1,iii})=3;

id_cueL{1,iii}=find(loaded_cueinput{1,iii}.CueL==1);
id_cueC{1,iii}=find(loaded_cueinput{1,iii}.CueC==1);
id_cueR{1,iii}=find(loaded_cueinput{1,iii}.CueR==1);

cue_ips{1,iii}(id_cueL{1,iii})= 1; 
cue_ips{1,iii}(id_cueC{1,iii})= 2; 
cue_ips{1,iii}(id_cueR{1,iii})= 3; 
end

for k = 1:length(actions)
    for kk = 1:length(actions{1,k})
    cue_prob{1,k}{1,kk} = antibias_rule(actions{1,k}(1:kk));
    end 
end
for k = 1:length(cue_prob)
    cue_mat{1,k} = cell2mat(cue_prob{1,k}');
end
%remove nan....
cueips=cue_ips;
cuemat=cue_mat;
actions2=actions;
for k=1:length(cue_ips)
remnanc{1,k}=find(isnan(cue_ips{1,k}));
remnanac{1,k}=find(isnan(actions{1,k}));
cueips{1,k}(isnan(cueips{1,k}))=[];
cuemat{1,k}(remnanc{1,k},:)=[];
actions2{1,k}(remnanc{1,k},:)=[];
end
%remove these

% val_for_lever_cued = nan(length(cue_ips),1);
for tt= 1:length(cueips)
for t = 1:length(cueips{1,tt})-1
    val_for_lever_cued{1,tt}(1,t) = cuemat{1,tt}(t,cueips{1,tt}(t+1));
end
end

%this val_for_lever_cued --- what was the probability (or value) of lever
%that was cued

%what was the probability that cued lever will be cued (based on AB)
for tt= 1:length(cueips)
for t = 1:length(cueips{1,tt})-1
    val_for_lever_actions{1,tt}(1,t) = cuemat{1,tt}(t,actions2{1,tt}(t+1));
end
end
valact = cell2mat(val_for_lever_actions);
valcue= cell2mat(val_for_lever_cued);
%this val_for_lever_actions ---what was the probability (or value) of lever
%that was pressed to give reward based on antibiasing rule 

%what was the probability that the action that animal selected would be
%rewarded based on antibiaing rule ?? 
%% 

chunks_100 = round(length(valact)./100);

for n =1:chunks_100
    if n < chunks_100
        meanval(1,n) = mean(valact(100*n-99:100*n));
    elseif n == chunks_100
        meanval(1,n) = mean(valact(100*n-99:end));
    end
end

for n =1:chunks_100
    if n < chunks_100
        meanvalcue(1,n) = mean(valcue(100*n-99:100*n));
    elseif n == chunks_100
        meanvalcue(1,n) = mean(valcue(100*n-99:end));
    end
end

figure(1)
plot(meanval,'b')
hold on 
plot(meanvalcue,'r')
ylim([0 1])
hold on 
yline(0.33,'--')
legend('probablity that her action gets her reward given AB rule','probabity of cued lever acc to AB rule')

%%%join the cuemat in excels files... 
cd ('W:\Lab\dms_lesion\data_modelver4\csv_t5tiffany_control')
for j = 1:length(loaded_cueinput)
newtables{1,j}=loaded_cueinput{1,j}.Variables;
newalltab{1,j}=[newtables{1,j},cue_mat{1,j}];
end

factor_input = {'CueL','CueC','CueR','Rew_L_3','Rew_L_2','Rew_L_1','Rew_C_3','Rew_C_2','Rew_C_1','Rew_R_3','Rew_R_2','Rew_R_1','nonRew_L_3','nonRew_L_2','nonRew_L_1','nonRew_C_3','nonRew_C_2','nonRew_C_1','nonRew_R_3','nonRew_R_2','nonRew_R_1','bias','AB_L','AB_C','AB_R'};
factorip_count = length(factor_input);
for j = 1:length(loaded_cueinput)
newinputs{1,j}=array2table(newalltab{1,j},'VariableNames',factor_input);
end
cd ('W:\Lab\dms_lesion\data_modelver5\csv_t5tiffany_control\')
for n = 1:length(newinputs)
writetable(newinputs{1,n},['inputtable_session_',num2str(n),'.csv']);
end


%GO TO SPYDER, GET AR
