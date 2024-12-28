 clear all; close all;
%% 
L = 2000;
p = 5;
q = 10;

lamda_RLS = 0.99;%RLS
lamda_RLS_test = 0.99;%此时不能太大
lamda_New = 0.99;%此时不能太大
lamda_New1 = 0.99;%此时不能太大
lamda_New2 = 0.99;%此时不能太大
sigma_MCC= 5;%%MCC

orders=4;
orders1=8;
orders2=16;
%%  噪声参数的设置
%% 噪声
% 均值
miu_real1=[-2.5 2.5]; 
miu_real2=[0 0]; 

miu_real1_4=[-3.5  -2    2    3.5]; 
miu_real2_4=[0 0 0 0]; 
% 标准差
sigma_real1=[1.6 1.6];
sigma_real2=[1.6 21];

sigma_real1_4=[ 1   1.5    1.5   1];
sigma_real2_4=[0.5 1.7 4.8 33.3];
% sigma_real=[0.5 0.5 10 10];
% 权重
weight_real1=[0.49 0.51];
weight_real2=[0.98 0.02];



weight_real1_4=[0.05   0.45   0.45   0.05];
weight_real2_4=[0.36 0.56 0.07 0.01];
 
   %%  
  MIU_Rand=[miu_real1(1)+j*miu_real2(1) miu_real1(1)+j*miu_real2(2)  miu_real1(2)+j*miu_real2(1)  miu_real1(2)+j*miu_real2(2)];
  SIGMA_Rand=[sqrt(sigma_real1(1)^2+sigma_real2(1)^2) sqrt(sigma_real1(1)^2+sigma_real2(2)^2) sqrt(sigma_real1(2)^2+sigma_real2(1)^2) sqrt(sigma_real1(2)^2+sigma_real2(2)^2)];
  WEIGHT_Rand=[weight_real1(1)*weight_real2(1) weight_real1(1)*weight_real2(2) weight_real1(2)*weight_real2(1) weight_real1(2)*weight_real2(2)];

 MIU_Rand1=[miu_real1(1)+j*miu_real2_4(1) miu_real1(1)+j*miu_real2_4(2) miu_real1(1)+j*miu_real2_4(3) miu_real1(1)+j*miu_real2_4(4) miu_real1(2)+j*miu_real2_4(1)  miu_real1(2)+j*miu_real2_4(2) miu_real1(2)+j*miu_real2_4(3)  miu_real1(2)+j*miu_real2_4(4)];
  SIGMA_Rand1=[sqrt(sigma_real1(1)^2+sigma_real2_4(1)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(2)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(3)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(4)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(1)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(2)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(3)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(4)^2)];
  WEIGHT_Rand1=[weight_real1(1)*weight_real2_4(1) weight_real1(1)*weight_real2_4(2) weight_real1(1)*weight_real2_4(3) weight_real1(1)*weight_real2_4(4) weight_real1(2)*weight_real2_4(1) weight_real1(2)*weight_real2_4(2) weight_real1(2)*weight_real2_4(3) weight_real1(2)*weight_real2_4(4)];


MIU_Rand2=[miu_real1_4(1)+j*miu_real2_4(1) miu_real1_4(1)+j*miu_real2_4(2) miu_real1_4(1)+j*miu_real2_4(3) miu_real1_4(1)+j*miu_real2_4(4) miu_real1_4(2)+j*miu_real2_4(1)  miu_real1_4(2)+j*miu_real2_4(2) miu_real1_4(2)+j*miu_real2_4(3)  miu_real1_4(2)+j*miu_real2_4(4) miu_real1_4(3)+j*miu_real2_4(1) miu_real1_4(3)+j*miu_real2_4(2) miu_real1_4(3)+j*miu_real2_4(3) miu_real1_4(3)+j*miu_real2_4(4) miu_real1_4(4)+j*miu_real2_4(1) miu_real1_4(4)+j*miu_real2_4(2) miu_real1_4(4)+j*miu_real2_4(3) miu_real1_4(4)+j*miu_real2_4(4)];
  SIGMA_Rand2=[sqrt(sigma_real1_4(1)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(4)^2)];
  WEIGHT_Rand2=[weight_real1_4(1)*weight_real2_4(1) weight_real1_4(1)*weight_real2_4(2) weight_real1_4(1)*weight_real2_4(3) weight_real1_4(1)*weight_real2_4(4) weight_real1_4(2)*weight_real2_4(1) weight_real1_4(2)*weight_real2_4(2) weight_real1_4(2)*weight_real2_4(3) weight_real1_4(2)*weight_real2_4(4) weight_real1_4(3)*weight_real2_4(1) weight_real1_4(3)*weight_real2_4(2) weight_real1_4(3)*weight_real2_4(3) weight_real1_4(3)*weight_real2_4(4) weight_real1_4(4)*weight_real2_4(1) weight_real1_4(4)*weight_real2_4(2) weight_real1_4(4)*weight_real2_4(3) weight_real1_4(4)*weight_real2_4(4)];


aa=0;
% bb=0;
for k=1:orders
    aa=aa+  WEIGHT_Rand(k)/ SIGMA_Rand(k)^2;
%       bb=bb+(sigma_real(k))*weight_real(k)*miu_real(k)/sigma_real(k)^2;
end

aa1=0;
for k=1:orders1
    aa1=aa1+  WEIGHT_Rand1(k)/ SIGMA_Rand1(k)^2;
%       bb=bb+(sigma_real(k))*weight_real(k)*miu_real(k)/sigma_real(k)^2;
end

aa2=0;
for k=1:orders2
    aa2=aa2+  WEIGHT_Rand2(k)/ SIGMA_Rand2(k)^2;
%       bb=bb+(sigma_real(k))*weight_real(k)*miu_real(k)/sigma_real(k)^2;
end
tic
for mm = 1:20
      %% 混合高斯噪声
    VV=zeros(1,L);
%      p_rand1 = rand(1, L);
% 
%     for n=1:L
%         if p_rand1(n)<weight_real1(1)
%             VV1(n)=normrnd(miu_real1(1),sigma_real1(1));
%         else
%             VV1(n)=normrnd(miu_real1(2),sigma_real1(2));
%         end
%     end
VV1=rand(1,L)*10-5;

% VV1=alpha_noise(1.7,0,1,0,1,L);
% VV1=randn(1,L);
% 
%  p_rand2 = rand(1, L);
%         for n=1:L
%         if p_rand2(n)<weight_real2(1)
%             VV2(n)=normrnd(miu_real2(1),sigma_real2(1));
%         else
%             VV2(n)=normrnd(miu_real2(2),sigma_real2(2));
%         end
%     end
%% 瑞利
%         vv=raylrnd(8,1,L);
%% 均匀分布
%   vv=2*rand(1,L)-1;
% VV2=randn(1,L)*1;
%     vv=VV-mean(VV);
VV2=alpha_noise(1.7,0,1,0,1,L);
% VV2=randn(1,L);
vv=VV1+VV2.*j;
% vv=randn(1,L)+j*randn(1,L);



     wo0 = randn(p,1)+j*randn(p,1);
%      wo00= randn(p,1)+j*randn(p,1);
wo1=wo0;
     wo = [ kron(wo1, ones(1,L)) ];

    uu = randn(p,L)+j*randn(p,L);
    for ii = 1 : L
        dd(ii) = (wo(:,ii))' * uu(:,ii) + vv(ii);
    end

    w_LMS = zeros(p,1)+j*zeros(p,1);   
    w_RLS = w_LMS;
       w_RLS_test = w_LMS;

           w_MCC = w_LMS;
 
    w_RLS_New = w_LMS;
 w_RLS_New1 = w_LMS;
 w_RLS_New2 = w_LMS;
    %% LMS
    mu_LMS = 0.01;
    for ii = 1 : L
        Err_LMS(mm,ii) = (wo(:,ii) - w_LMS)' *  (wo(:,ii) - w_LMS);

        dn = dd(ii);
        un = uu(:,ii);
        en = dn - w_LMS' * un;
        w_LMS = w_LMS +  mu_LMS * conj(en) * un;

    end


    % RLS_MSE
    Pn = eye(p)*1;
    for ii = 1 : L
        Err_RLS(mm,ii) = (wo(:,ii)  - w_RLS)' * (wo(:,ii)  - w_RLS);
        dn = dd(ii);
     un = [uu(:,ii)];
        en = dn - (w_RLS)' * un;

          kn = Pn * un / ( lamda_RLS+ (un)' * Pn * un );
        Pn = 1/lamda_RLS * ( Pn - kn *(un)' * Pn);
        w_RLS = w_RLS +kn * conj(en);
    end
    


    

      %% MCC
       mu_MCC=0.02;
          for ii = 1 : L
                 Err_MCC(mm,ii) = (wo(:,ii)  - w_MCC)' * (wo(:,ii)  - w_MCC);
                dn = dd(ii);
       un = [uu(:,ii)];

        ek_MCC = dn- w_MCC'*un;

        w_MCC = w_MCC + mu_MCC * exp(-(abs(ek_MCC))^2/(2*sigma_MCC^2))*conj(ek_MCC)*un ;
          end

        % RLS_New
    Pn = eye(p)*1;
    for i = 1 : L
        Err_RLS_New(mm,i) = (wo(:,i)  - w_RLS_New)' * (wo(:,i)  - w_RLS_New);
        dn = dd(i);
    un = uu(:,i);
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


       % RLS_New1
    Pn1 = eye(p)*1;
    for i = 1 : L
        Err_RLS_New1(mm,i) = (wo(:,i)  - w_RLS_New1)' * (wo(:,i)  - w_RLS_New1);
        dn = dd(i);
    un = uu(:,i);
        ek_Rand1 = dn - (w_RLS_New1)' * un;
         for k=1:orders1
                 P_Rand1(i,k)=exp(-1*abs(ek_Rand1-MIU_Rand1(k))^2/(SIGMA_Rand1(k)^2))/(pi*SIGMA_Rand1(k)^2);
            end
            for k=1:orders1
                V_Rand1(i,k)=WEIGHT_Rand1(k)*P_Rand1(i,k)/(WEIGHT_Rand1*P_Rand1(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders1
                R1=R1+V_Rand1(i,k)/SIGMA_Rand1(k)^2;
                  R2=R2+V_Rand1(i,k)*conj(MIU_Rand1(k))/SIGMA_Rand1(k)^2;
            end
          kn1 = Pn1 * un / ( lamda_New1+ R1*(un)' * Pn1 * un );
        Pn1 = 1/lamda_New1 * ( Pn1 - R1*kn1 * (un)' * Pn1);
        w_RLS_New1 = w_RLS_New1 +kn1 * (R1*conj(ek_Rand1)-R2);
    end

 % RLS_New2
    Pn2 = eye(p)*1;
    for i = 1 : L
        Err_RLS_New2(mm,i) = (wo(:,i)  - w_RLS_New2)' * (wo(:,i)  - w_RLS_New2);
        dn = dd(i);
    un = uu(:,i);
        ek_Rand2 = dn - (w_RLS_New2)' * un;
         for k=1:orders2
                P_Rand2(i,k)=exp(-1*abs(ek_Rand2-MIU_Rand2(k))^2/(SIGMA_Rand2(k)^2))/(pi*SIGMA_Rand2(k)^2);
            end
            for k=1:orders2
                V_Rand2(i,k)=WEIGHT_Rand2(k)*P_Rand2(i,k)/(WEIGHT_Rand2*P_Rand2(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders2
                R1=R1+V_Rand2(i,k)/SIGMA_Rand2(k)^2;
                  R2=R2+V_Rand2(i,k)*conj(MIU_Rand2(k))/SIGMA_Rand2(k)^2;
            end
        kn2 = Pn2 * un / ( lamda_New2+ R1*(un)' * Pn2 * un );
        Pn2 = 1/lamda_New2 * ( Pn2 - R1*kn2 * (un)' * Pn2);
        w_RLS_New2 = w_RLS_New2 +kn2 * (R1*conj(ek_Rand2)-R2);
    end

       Err_TH_RLS_New(mm) = (1-lamda_New)*p/ mean(diag(uu*uu'/L))/aa/ (1+lamda_New);   
       Err_TH_RLS_New_2(mm) = (1-lamda_New1)*p/ mean(diag(uu*uu'/L))/aa1/ (1+lamda_New1);   
       Err_TH_RLS_New_3(mm) = (1-lamda_New2)*p/ mean(diag(uu*uu'/L))/aa2/ (1+lamda_New2);   
end
toc

gap=80;
x= 1:gap:L;
figure,hold on;
Err1=mean(Err_LMS);

Err3=mean(Err_RLS);

Err4=mean(Err_MCC);
% Err5=mean(Err_RLS_New);
Err6=mean(Err_RLS_New);
Err7=mean(Err_RLS_New1);
Err8=mean(Err_RLS_New2);


plot(x,10* log10(Err1(1:gap:L)),'*','MarkerSize',10,'LineWidth',1,'Color','g');
% plot(x,10* log10(Err2(1:gap:L)),'p','MarkerSize',10,'LineWidth',1,'Color','m');
plot(x,10* log10(Err3(1:gap:L)),'h','MarkerSize',10,'LineWidth',1,'Color','b');
plot(x,10* log10(Err4(1:gap:L)),'s','MarkerSize',10,'LineWidth',1,'Color','r');
plot(x,10* log10(Err6(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','c');

plot(x,10* log10(Err7(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','m');
plot(x,10* log10(Err8(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','y');

plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New)),'c--','linewidth',2.5)
plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New_2)),'m--','linewidth',2.5)
plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New_3)),'y--','linewidth',2.5)

plot(10* log10(mean(Err_LMS)),'g','LineWidth',1);

plot(10*log10(mean(Err_RLS)),'b','LineWidth',1);
plot(10* log10(mean(Err_MCC)),'r','LineWidth',1);

plot(10*log10(mean(Err_RLS_New)),'c','LineWidth',1),

plot(10*log10(mean(Err_RLS_New1)),'m','LineWidth',1),
plot(10*log10(mean(Err_RLS_New2)),'y','LineWidth',1),

legend('CLMS (\eta=0.01)','CRLS (\lambda=0.99)','MCCC (\eta=0.02,\sigma=5)','RMCL (K_R=2,K_I=2,\lambda=0.99)','RMCL (K_R=2,K_I=4,\lambda=0.99)','RMCL (K_R=4,K_I=4,\lambda=0.99)','TH (RMCL,K_R=2,K_I=2,\lambda=0.99)','TH (RMCL,K_R=2,K_I=4,\lambda=0.99)','TH (RMCL,K_R=4,K_I=4,\lambda=0.99)');
xlabel('Iterations');ylabel('MSD(dB)');
box on;
grid on;
xlim([0,L]);
ylim([-16,35]);




