clear all; close all;
L = 2000; %%Signal Length
p = 5;    %% Filter Size(Linear Model)
q = 10;  %%Double Filter Size(Widely Linear Model)

lamda_RLS = 0.99; %WL-RLS
lamda_RLS_test = 0.99;
lamda_New = 0.99;

lamda_New1 = 0.99;
sigma_MCC= 2;%%WL-MCCC

orders=4;
%%  噪声参数的设置
%% 脉冲噪声
% 均值
miu_real1=[0 0];
miu_real2=[-2.5 2.5];

% 标准差
sigma_real1=[0.5 20];
sigma_real2=[1.5 1.5];

% 权重
weight_real1=[0.95 0.05];
weight_real2=[0.5 0.5];


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
for mm = 1:500
    %% 混合高斯噪声
    p_rand1 = rand(1, L);

    for n=1:L
        if p_rand1(n)<weight_real1(1)
            VV1(n)=normrnd(miu_real1(1),sigma_real1(1));
        else
            VV1(n)=normrnd(miu_real1(2),sigma_real1(2));
        end
    end
    %% 均匀分布

    VV2=rand(1,L)*10-5;
    vv=VV1+VV2.*j;
    wo0 = randn(p,1)+j*randn(p,1);
    wo00= randn(p,1)+j*randn(p,1);
    wo1=[wo0; wo00];
    wo = [ kron(wo1, ones(1,L)) ];
    
    %% Generate input signals (e.g., QPSK modulated signals)
    data = randi([0 3], L, 5);
    s = pskmod(data, 4, pi/4);  % QPSK modulation
    uu=s.';

    for ii = 1 : L
        dd(mm,ii) = (wo(:,ii))' * [uu(:,ii);conj(uu(:,ii))] + vv(ii);
        y(mm,ii)= (wo(:,ii))' * [uu(:,ii);conj(uu(:,ii))];
    end

    w_LMS = zeros(2*p,1)+j*zeros(2*p,1);
    w_RLS = w_LMS;
    w_RLS_test = w_LMS;
    w_MCC = w_LMS;
    w_RLS_New1 = w_LMS;

    %% LMS
    mu_LMS = 0.02;
    for ii = 1 : L
        Err_LMS(mm,ii) = (wo(:,ii) - w_LMS)' *  (wo(:,ii) - w_LMS);
             dn = dd(mm,ii);
        un = [uu(:,ii);conj(uu(:,ii))];
        y_LMS(mm,ii)=w_LMS' * un;
        en_LMS(mm,ii) = dn - w_LMS' * un;
        w_LMS = w_LMS +  mu_LMS * conj(en_LMS(mm,ii)) * un;
    end
    %% RLS
    Cn = 1e-0*eye(p)*1;
    Dn = 2e-1*eye(p)*1;
    for ii = 1 : L
        Err_RLS_test(mm,ii) = (wo(:,ii)  - w_RLS_test)' * (wo(:,ii)  - w_RLS_test);
        dn = dd(mm,ii);
        un = [uu(:,ii);conj(uu(:,ii))];
        y_RLS(mm,ii)= (w_RLS_test)'*un;
        en_RLS(mm,ii) = dn -  (w_RLS_test)'*un;
        Hn=Cn*uu(:,ii)+Dn*conj(uu(:,ii));
        kn_test=1/( lamda_RLS_test+2*real(uu(:,ii)'*Hn))*Hn;
        Cn=1/lamda_RLS_test*(Cn-Hn*kn_test');
        Dn=1/lamda_RLS_test*(Dn-Hn*kn_test.');
        w_RLS_test = w_RLS_test +[kn_test;conj(kn_test)] * conj(en_RLS(mm,ii));
    end
    %% WL-MCCC
    mu_MCC=0.04;
    for ii = 1 : L
        Err_MCC(mm,ii) = (wo(:,ii)  - w_MCC)' * (wo(:,ii)  - w_MCC);
        dn = dd(mm,ii);
        un = [uu(:,ii);conj(uu(:,ii))];
        y_MCC(mm,ii)=w_MCC'*un;
        ek_MCC(mm,ii) = dn- w_MCC'*un;

        w_MCC = w_MCC + mu_MCC * exp(-(abs(ek_MCC(mm,ii)))^2/(2*sigma_MCC^2))*conj(ek_MCC(mm,ii))*un ;
    end
    %% WL-RMCL
    Cn = 1e-0*eye(p)*1;
    Dn = 2e-1*eye(p)*1;
    for i = 1 : L
        Err_RLS_New1(mm,i) = (wo(:,i)  - w_RLS_New1)' * (wo(:,i)  - w_RLS_New1);
        dn = dd(mm,i);
        un = [uu(:,i);conj(uu(:,i))];
        y_RLS_New(mm,i)=(w_RLS_New1)' * un;
        ek_Rand1(mm,i) = dn - (w_RLS_New1)' * un;
        for k=1:orders
               P_Rand1(i,k)=exp(-1*abs(ek_Rand1(mm,i)-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
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
        w_RLS_New1 = w_RLS_New1+[kn1;conj(kn1)] * (R1*conj(ek_Rand1(mm,i))-R2);

    end

    Err_TH_MSE(mm) = (1-lamda_RLS)/(1+lamda_RLS) * p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L)) * var(vv) / 1;
    Err_TH_RLS_New(mm) = (1-lamda_New)*p*2/ mean(diag([uu;conj(uu)]*[uu;conj(uu)]'/L))/aa/ (1+lamda_New);
end
toc


gap=60;
x= 1:gap:L;
Received=mean(real(dd));
Clean=mean(real(y));
LMS_Signal=mean(real(y_LMS));
MCC_Signal=mean(real(y_MCC));
RLS_Signal=mean(real(y_RLS));
RMCL_Signal=mean(real(y_RLS_New));

Received1=mean(imag(dd));
Clean1=mean(imag(y));
LMS_Signal1=mean(imag(y_LMS));
MCC_Signal1=mean(imag(y_MCC));
RLS_Signal1=mean(imag(y_RLS));
RMCL_Signal1=mean(imag(y_RLS_New));


Received2=mean((dd));
Clean2=mean(y);
LMS_Signal2=mean((y_LMS));
MCC_Signal2=mean((y_MCC));
RLS_Signal2=mean((y_RLS));
RMCL_Signal2=mean((y_RLS_New));



figure
subplot(211),hold on
plot(x,10*log10(abs(Received(1:gap:L))),'r--','LineWidth',1);
plot(x,10*log10(abs(Clean(1:gap:L))),'m--','LineWidth',1);
plot(x,10*log10(abs(LMS_Signal(1:gap:L))),'k--','LineWidth',1);
plot(x,10*log10(abs(MCC_Signal(1:gap:L))),'c--','LineWidth',1);
plot(x,10*log10(abs(RLS_Signal(1:gap:L))),'g--','LineWidth',1);
plot(x,10*log10(abs(RMCL_Signal(1:gap:L))),'b--','LineWidth',1);
xlabel('Time Index');ylabel('Magnitude(dB)');

box on;
grid on;
xlim([L/2,L]);
title('Real Part');

subplot(212),hold on
plot(x,10*log10(abs(Received1(1:gap:L))),'r--','LineWidth',1);
plot(x,10*log10(abs(Clean1(1:gap:L))),'m--','LineWidth',1);
plot(x,10*log10(abs(LMS_Signal1(1:gap:L))),'k--','LineWidth',1);
plot(x,10*log10(abs(MCC_Signal1(1:gap:L))),'c--','LineWidth',1);
plot(x,10*log10(abs(RLS_Signal1(1:gap:L))),'g--','LineWidth',1);
plot(x,10*log10(abs(RMCL_Signal1(1:gap:L))),'b--','LineWidth',1);
xlabel('Time Index');ylabel('Magnitude(dB)');
legend('Receiverd Signal','Clean Signal','Equalized Signal 1(ACLMS)','Equalized Signal 2(WL-MCCC)','Equalized Signal 3(WL-RLS)','Equalized Signal 4(WL-RMCL)');
box on;
grid on;
title('Imagniary Part');
xlim([L/2,L]);

E1=mean(abs(en_LMS));
E2=mean(abs(ek_MCC));
E3=mean(abs(en_RLS));
E4=mean(abs(ek_Rand1));

figure,hold on;
plot(x,10* log10(E1(1:gap:L)),'k','LineWidth',1);
plot(x,10* log10(E2(1:gap:L)),'c','LineWidth',1);
plot(x,10* log10(E3(1:gap:L)),'g','LineWidth',1);
plot(x,10* log10(E4(1:gap:L)),'b','LineWidth',1),
legend('ACLMS (\eta=0.02)','WL-MCCC (\eta=0.035,\sigma=2)','WL-RLS (\lambda=0.99)','WL-RMCL (\lambda=0.99)');
xlabel('Time Index');ylabel('Error(dB)');
box on;
grid on;
xlim([0,L]);

