
t = readtable('C:/Users/MSI GF63/Desktop/sig_feat_p2mm.csv');
g = table2array(t(:,2));
Features=table2array(t(:,3:end));

for i=3:(length(Features)+1)
    c = (t(1,i));
    k = table2array(t(:,i));
    [p,tbl] = anova1(k,g);
    pvalue = tbl{2,6};
    Fstat = tbl{2,5};
    c
    disp(['La valeur de pvalue de  vaut : ' num2str(pvalue)])
    disp(['La valeur de fstat de  vaut : ' num2str(Fstat)])
end 
    
