 clear all; close all;
%% 
L = 2000;
p = 5;
q = 10;

lamda_RLS_test = 0.99;%此时不能太大
lamda_New = 0.99;%此时不能太大
lamda_New1 = 0.99;%此时不能太大
lamda_New2 = 0.99;%此时不能太大
lamda_New3 = 0.99;%此时不能太大

sigma_MCC= 5;%%MCC

orders=4;
orders1=8;
orders2=16;
%%  噪声参数的设置
% 均值
miu_real1=[-2.5 2.5]; 
miu_real2=[0 0]; 

miu_real1_4=[-4.1  -2.3    2.3    4.1]; 
miu_real2_4=[0 0 0 0]; 
% 标准差
sigma_real1=[1.6 1.6];
sigma_real2=[1.6 21];

sigma_real1_4=[ 1   1.5    1.5   1];
sigma_real2_4=[0.5 1.5 5 34];
% sigma_real=[0.5 0.5 10 10];
% 权重
weight_real1=[0.49 0.51];
weight_real2=[0.98 0.02];


weight_real1_4=[0.05   0.45   0.45  0.05];
weight_real2_4=[0.36 0.56 0.07 0.01];
 
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
% VV1=rand(1,L)*4-2;
VV1=rand(1,L)*10-5;


%  p_rand2 = rand(1, L);
%         for n=1:L
%         if p_rand2(n)<weight_real2(1)
%             VV2(n)=normrnd(miu_real2(1),sigma_real2(1));
%         else
%             VV2(n)=normrnd(miu_real2(2),sigma_real2(2));
%         end
%     end
VV2=alpha_noise(1.7,0,1,0,1,L);
%% 瑞利
%         vv=raylrnd(8,1,L);
%% 均匀分布
%   vv=2*rand(1,L)-1;
% VV2=randn(1,L)*1;
%     vv=VV-mean(VV);
vv=VV1+VV2.*j;
% vv=randn(1,L)+j*randn(1,L);


     wo0 = randn(p,1)+j*randn(p,1);
     wo00= randn(p,1)+j*randn(p,1);
wo1=[wo0; wo00];
     wo = [ kron(wo1, ones(1,L)) ];

    uu = randn(p,L)+j*randn(p,L);
    for ii = 1 : L
        dd(ii) = (wo(:,ii))' * [uu(:,ii);conj(uu(:,ii))] + vv(ii);
    end

    w_LMS = zeros(2*p,1)+j*zeros(2*p,1);   
    w_RLS = w_LMS;
       w_RLS_test = w_LMS;

           w_MCC = w_LMS;
  

    w_RLS_New = w_LMS;
 w_RLS_New1 = w_LMS;
 w_RLS_New2 = w_LMS;
 w_RLS_New3 = w_LMS;
    %% LMS
    mu_LMS = 0.01;
    for ii = 1 : L
        Err_LMS(mm,ii) = (wo(:,ii) - w_LMS)' *  (wo(:,ii) - w_LMS);

        dn = dd(ii);
        un = [uu(:,ii);conj(uu(:,ii))];
        en = dn - w_LMS' * un;
        w_LMS = w_LMS +  mu_LMS * conj(en) * un;

    end


        %% RLS_TEST

    Cn = 1e-0*eye(p)*1;
   Dn = 2e-1*eye(p)*1;
    for ii = 1 : L
        Err_RLS_test(mm,ii) = (wo(:,ii)  - w_RLS_test)' * (wo(:,ii)  - w_RLS_test);
        dn = dd(ii);
     un = [uu(:,ii);conj(uu(:,ii))];
        en = dn -  (w_RLS_test)'*un;
Hn=Cn*uu(:,ii)+Dn*conj(uu(:,ii));
kn_test=1/( lamda_RLS_test+2*real(uu(:,ii)'*Hn))*Hn;
Cn=1/lamda_RLS_test*(Cn-Hn*kn_test');
Dn=1/lamda_RLS_test*(Dn-Hn*kn_test.');
   
        w_RLS_test = w_RLS_test +[kn_test;conj(kn_test)] * conj(en);
    end
    

      %% MCC
       mu_MCC=0.02;
          for ii = 1 : L
                 Err_MCC(mm,ii) = (wo(:,ii)  - w_MCC)' * (wo(:,ii)  - w_MCC);
                dn = dd(ii);
       un = [uu(:,ii);conj(uu(:,ii))];

        ek_MCC = dn- w_MCC'*un;

        w_MCC = w_MCC + mu_MCC * exp(-(abs(ek_MCC))^2/(2*sigma_MCC^2))*conj(ek_MCC)*un ;
          end


            %% RLS_New
    Cn = 1e-0*eye(p)*1;
     Dn = 2e-1*eye(p)*1;
    for i = 1 : L
        Err_RLS_New1(mm,i) = (wo(:,i)  - w_RLS_New1)' * (wo(:,i)  - w_RLS_New1);
        dn = dd(i);
    un = [uu(:,i);conj(uu(:,i))];
        ek_Rand1 = dn - (w_RLS_New1)' * un;
         for k=1:orders
                 P_Rand1(i,k)=exp(-1*abs(ek_Rand1-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
            end
            for k=1:orders
                V_Rand1(i,k)=WEIGHT_Rand(k)*P_Rand1(i,k)/(WEIGHT_Rand*P_Rand1(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders
                R1=R1+V_Rand1(i,k)/SIGMA_Rand(k)^2;
                  R2=R2+V_Rand1(i,k)*conj(MIU_Rand(k))/SIGMA_Rand(k)^2;
            end


            Hn=Cn*uu(:,i)+Dn*conj(uu(:,i));
            kn1= 1/(lamda_New1+2*R1*real(uu(:,i)'*Hn))*Hn;
Cn=1/  lamda_New1*(Cn-R1*Hn*kn1');
Dn=1/  lamda_New1*(Dn-R1*Hn*kn1.');
w_RLS_New1 = w_RLS_New1+[kn1;conj(kn1)] * (R1*conj(ek_Rand1)-R2);

    

    end


         %% RLS_New_2
    Cn1 = 1e-0*eye(p)*1;
     Dn1 = 2e-1*eye(p)*1;
    for i = 1 : L
        Err_RLS_New2(mm,i) = (wo(:,i)  - w_RLS_New2)' * (wo(:,i)  - w_RLS_New2);
        dn = dd(i);
    un = [uu(:,i);conj(uu(:,i))];
        ek_Rand2 = dn - (w_RLS_New2)' * un;
         for k=1:orders1

                 P_Rand2(i,k)=exp(-1*abs(ek_Rand2-MIU_Rand1(k))^2/(SIGMA_Rand1(k)^2))/(pi*SIGMA_Rand1(k)^2);
            end
            for k=1:orders1
                V_Rand2(i,k)=WEIGHT_Rand1(k)*P_Rand2(i,k)/(WEIGHT_Rand1*P_Rand2(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders1
                R1=R1+V_Rand2(i,k)/SIGMA_Rand1(k)^2;
                  R2=R2+V_Rand2(i,k)*conj(MIU_Rand1(k))/SIGMA_Rand1(k)^2;
            end


            Hn1=Cn1*uu(:,i)+Dn1*conj(uu(:,i));
            kn2= 1/(lamda_New2+2*R1*real(uu(:,i)'*Hn1))*Hn1;
Cn1=1/  lamda_New2*(Cn1-R1*Hn1*kn2');
Dn1=1/  lamda_New2*(Dn1-R1*Hn1*kn2.');
w_RLS_New2= w_RLS_New2+[kn2;conj(kn2)] * (R1*conj(ek_Rand2)-R2);

    

    end

    %% RLS_New_3
    Cn2 = 1e-0*eye(p)*1;
     Dn2 = 2e-1*eye(p)*1;
    for i = 1 : L
        Err_RLS_New3(mm,i) = (wo(:,i)  - w_RLS_New3)' * (wo(:,i)  - w_RLS_New3);
        dn = dd(i);
    un = [uu(:,i);conj(uu(:,i))];
        ek_Rand3 = dn - (w_RLS_New3)' * un;
         for k=1:orders2

                 P_Rand3(i,k)=exp(-1*abs(ek_Rand3-MIU_Rand2(k))^2/(SIGMA_Rand2(k)^2))/(pi*SIGMA_Rand2(k)^2);
            end
            for k=1:orders2
                V_Rand3(i,k)=WEIGHT_Rand2(k)*P_Rand3(i,k)/(WEIGHT_Rand2*P_Rand3(i,:)');
            end
         R1=0;
         R2=0;
            for k=1:orders2
                R1=R1+V_Rand3(i,k)/SIGMA_Rand2(k)^2;
                  R2=R2+V_Rand3(i,k)*conj(MIU_Rand2(k))/SIGMA_Rand2(k)^2;
            end


            Hn2=Cn2*uu(:,i)+Dn2*conj(uu(:,i));
            kn3= 1/(lamda_New3+2*R1*real(uu(:,i)'*Hn2))*Hn2;
Cn2=1/  lamda_New3*(Cn2-R1*Hn2*kn3');
Dn2=1/  lamda_New3*(Dn2-R1*Hn2*kn3.');
w_RLS_New3= w_RLS_New3+[kn3;conj(kn3)] * (R1*conj(ek_Rand3)-R2);

    

    end


       Err_TH_RLS_New(mm) = (1-lamda_New1)*p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L))/aa/ (1+lamda_New1);   
              Err_TH_RLS_New_2(mm) = (1-lamda_New2)*p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L))/aa1/ (1+lamda_New2);   
                    Err_TH_RLS_New_3(mm) = (1-lamda_New3)*p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L))/aa2/ (1+lamda_New3);   
end
toc

gap=80;
x= 1:gap:L;
figure,hold on;
Err1=mean(Err_LMS);

Err3=mean(Err_RLS_test);
Err4=mean(Err_MCC);

Err6=mean(Err_RLS_New1);
Err7=mean(Err_RLS_New2);
Err8=mean(Err_RLS_New3);
plot(x,10* log10(Err1(1:gap:L)),'*','MarkerSize',10,'LineWidth',1,'Color','g');

plot(x,10* log10(Err3(1:gap:L)),'h','MarkerSize',10,'LineWidth',1,'Color','b');
plot(x,10* log10(Err4(1:gap:L)),'s','MarkerSize',10,'LineWidth',1,'Color','r');
plot(x,10* log10(Err6(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','c');
plot(x,10* log10(Err7(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','m');
plot(x,10* log10(Err8(1:gap:L)),'d','MarkerSize',10,'LineWidth',1,'Color','y');
plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New)),'c--','linewidth',2.5)
plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New_2)),'m--','linewidth',2.5)
plot(10*log10(ones(1,L)*mean(Err_TH_RLS_New_3)),'y--','linewidth',2.5)

plot(10* log10(mean(Err_LMS)),'g','LineWidth',1);

plot(10*log10(mean(Err_RLS_test)),'b','LineWidth',1);
plot(10* log10(mean(Err_MCC)),'r','LineWidth',1);

plot(10*log10(mean(Err_RLS_New1)),'c','LineWidth',1);
plot(10*log10(mean(Err_RLS_New2)),'m','LineWidth',1);
plot(10*log10(mean(Err_RLS_New3)),'y','LineWidth',1);
legend('ACLMS (\eta=0.01)','WL-RLS (\lambda=0.99)','WL-MCCC (\eta=0.02,\sigma=5)','WL-RMCL (K_R=2,K_I=2,\lambda=0.99)','WL-RMCL (K_R=2,K_I=4,\lambda=0.99)','WL-RMCL (K_R=4,K_I=4,\lambda=0.99)','TH (WL-RMCL,K_R=2,K_I=2,\lambda=0.99)','TH (WL-RMCL,K_R=2,K_I=4,\lambda=0.99)','TH (WL-RMCL,K_R=4,K_I=4,\lambda=0.99)');
xlabel('Iterations');ylabel('MSD(dB)');
box on;
grid on;
xlim([0,L]);
ylim([-12,40]);

