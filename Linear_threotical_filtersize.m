clear all; close all,clc;
%%
L = 2000;
p = 2;
q = p*2;
S=[];
Err_TH_RLS_New=[];
lamda_New=0.99;
orders=4;
%%  噪声参数的设置

%% 噪声
% 均值
miu_real1=[0 0];
miu_real2=[0 0];
% miu_real=[0 0 0 0];
% 标准差
sigma_real1=[1 2];
sigma_real2=[3 4];
% sigma_real=[0.5 0.5 10 10];
% 权重
weight_real1=[0.5 0.5];
weight_real2=[0.5 0.5];
% weight_real=[0.95 0.05];

%%
MIU_Rand=[miu_real1(1)+j*miu_real2(1) miu_real1(1)+j*miu_real2(2)  miu_real1(2)+j*miu_real2(1)  miu_real1(2)+j*miu_real2(2)];
SIGMA_Rand=[sqrt(sigma_real1(1)^2+sigma_real2(1)^2) sqrt(sigma_real1(1)^2+sigma_real2(2)^2) sqrt(sigma_real1(2)^2+sigma_real2(1)^2) sqrt(sigma_real1(2)^2+sigma_real2(2)^2)];
WEIGHT_Rand=[weight_real1(1)*weight_real2(1) weight_real1(1)*weight_real2(2) weight_real1(2)*weight_real2(1) weight_real1(2)*weight_real2(2)];

aa=0;
% bb=0;
for k=1:orders
    aa=aa+  WEIGHT_Rand(k)/ SIGMA_Rand(k)^2;
    %       bb=bb+(sigma_real(k))*weight_real(k)*miu_real(k)/sigma_real(k)^2;
end

tic
for kk=1:19
for mm=1:50
    %% 混合高斯噪声
    VV=zeros(1,L);

     p_rand1 = rand(1, L);

    for n=1:L
        if p_rand1(n)<weight_real1(1)
            VV1(n)=normrnd(miu_real1(1),sigma_real1(1));
        else
            VV1(n)=normrnd(miu_real1(2),sigma_real1(2));
        end
    end

 p_rand2 = rand(1, L);
        for n=1:L
        if p_rand2(n)<weight_real2(1)
            VV2(n)=normrnd(miu_real2(1),sigma_real2(1));
        else
            VV2(n)=normrnd(miu_real2(2),sigma_real2(2));
        end
    end

%     VV1=randn(1,L)*1;
%     VV2=randn(1,L)*2;
    vv=VV1+VV2.*j;



    wo0 = randn(p,1)+j*randn(p,1);
    wo00= randn(p,1)+j*randn(p,1);
    wo1=[wo0; wo00];
    wo = [ kron(wo1, ones(1,L)) ];

    uu = randn(p,L)+j*randn(p,L);
    for ii = 1 : L
        dd(ii) = (wo(:,ii))' * [uu(:,ii);conj(uu(:,ii))] + vv(ii);
    end

    w_RLS_New=zeros(2*p,1)+j*zeros(2*p,1);

       %% RLS_New
    Pn = eye(2*p)*1;
    for i = 1 : L
        Err_RLS_New(mm,i) = (wo(:,i)  - w_RLS_New)' * (wo(:,i)  - w_RLS_New);
        dn = dd(i);
    un = [uu(:,i);conj(uu(:,i))];
        ek_Rand = dn - (w_RLS_New)' * un;
         for k=1:orders
                 P_Rand(i,k)=exp(-1*abs(ek_Rand-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
            end
            for k=1:orders
                V_Rand(i,k)=WEIGHT_Rand(k)*P_Rand(i,k)/(WEIGHT_Rand*P_Rand(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders
                R1=R1+V_Rand(i,k)/SIGMA_Rand(k)^2;
                  R2=R2+V_Rand(i,k)*conj(MIU_Rand(k))/SIGMA_Rand(k)^2;
            end
          kn = Pn * un / ( lamda_New+ R1*(un)' * Pn * un );
        Pn = 1/lamda_New * ( Pn - R1*kn * (un)' * Pn);
        w_RLS_New = w_RLS_New +kn * (R1*conj(ek_Rand)-R2);
    end

end
    Err_TH_RLS_New=[Err_TH_RLS_New (1-lamda_New)*p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L))/aa/ (1+lamda_New)];
%     lamda_New=lamda_New+0.001;
p=p+1;
    S=[S mean(Err_RLS_New(:,end))];
disp(kk);
end
toc


x= 2:2:38;
figure(1),hold on;
% scatter(x,10*log10(S),'r-');
% scatter(x,10*log10(Err_TH_RLS_New),'c-');

plot(x,10*log10(S),'ro--','MarkerSize',8,'LineWidth',1);
plot(x, 10*log10(Err_TH_RLS_New),'cs--','MarkerSize',8,'LineWidth',1);
legend('Simulation (RMCL)','Theoretical Prediction (RMCL)');
xlabel('Filter Size');ylabel('SS-MSD(dB)');

box on;
grid on;
xlim([2,38]);
ylim([-8.5,2]);

