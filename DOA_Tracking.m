clear all,clc,close all;
M = 64; %天线阵子数

thetas1 = [60];%信号的真实波达角
thetas2 = [45];%信号的真实波达角
thetas3 = [30];%信号的真实波达角
lambda = 10;%这个随便设
d = lambda /2;%通常的假设
nfft  = 128;
cplen = 16;
nSym  = 10;%子载波数
nt    = 4;%天线数
N =(nfft+ cplen)*nSym*nt;%信号点数

KK=3;

sigma_MCC=1;

sigma_lms1=ones(KK,N*3);
mu_lms=zeros(KK,N*3);


% 生成方向矢量
A1 = exp(-1i * 2 * pi * d * (0:M-1)' * sind(thetas1) / lambda);%维度：天线阵子数*接收信号源。即对应多个信号源各自的steer vector
A2 = exp(-1i * 2 * pi * d * (0:M-1)' * sind(thetas2) / lambda);%维度：天线阵子数*接收信号源。即对应多个信号源各自的steer vector
A3 = exp(-1i * 2 * pi * d * (0:M-1)' * sind(thetas3) / lambda);%维度：天线阵子数*接收信号源。即对应多个信号源各自的steer vector


b1=1;

orders=4;
% 均值
miu_real1=[0 0]; 
miu_real2=[0 0]; 
% miu_real=[0 0 0 0]; 
% 标准差
sigma_real1=[0.5 20];
sigma_real2=[0.9 1.1];
% sigma_real=[0.5 0.5 10 10];
% 权重
weight_real1=[0.9 0.1];
weight_real2=[0.49 0.51];

  %%  
  MIU_Rand=[miu_real1(1)+j*miu_real2(1) miu_real1(1)+j*miu_real2(2)  miu_real1(2)+j*miu_real2(1)  miu_real1(2)+j*miu_real2(2)];
  SIGMA_Rand=[sqrt(sigma_real1(1)^2+sigma_real2(1)^2) sqrt(sigma_real1(1)^2+sigma_real2(2)^2) sqrt(sigma_real1(2)^2+sigma_real2(1)^2) sqrt(sigma_real1(2)^2+sigma_real2(2)^2)];
  WEIGHT_Rand=[weight_real1(1)*weight_real2(1) weight_real1(1)*weight_real2(2) weight_real1(2)*weight_real2(1) weight_real1(2)*weight_real2(2)];

for kk=1:KK
   
     VV1 = randn(M,N).*1i+alpha_noise(1.7,0,1,0,M,N).*1i;
    VV2 =randn(M,N).*1i+alpha_noise(1.7,0,1,0,M,N).*1i;
    VV3 = randn(M,N).*1i+alpha_noise(1.7,0,1,0,M,N).*1i;
    dataIn1 = complex(randn(nfft,nSym,nt),2*randn(nfft,nSym,nt));

    yy1 = ofdmmod(dataIn1,nfft,cplen);

    ofdm1=reshape(yy1,[],1);%由于传输需要


    n = 0:N-1;
    s = [ofdm1';...

    ];

% G1=rand(M,N);
% G2=rand(M,N);
% G3=rand(M,N);
% 
R1=rand(M,N);
R2=rand(M,N);
R3=rand(M,N);
TH=weight_real1(1);
%  VV1 = rand(M,N).*10-5+1i.*((R1>TH).*10.*randn(M,N)+(R1<TH).*0.5.*randn(M,N));
%     VV2 = rand(M,N).*10-5+1i.*((R2>TH).*10.*randn(M,N)+(R2<TH).*0.5.*randn(M,N));
%     VV3 =  rand(M,N).*10-5+1i.*((R3>TH).*10.*randn(M,N)+(R3<TH).*0.5.*randn(M,N));

% 
%     VV1 = 1i.*((G1>weight_real1(1)).*normrnd(miu_real1(1),sigma_real1(1),M,N)+(G1<weight_real1(1)).*normrnd(miu_real1(2),sigma_real1(2),M,N))+1*((R1>TH).*10.*randn(M,N)+(R1<TH).*0.5.*randn(M,N));
%     VV2 =1i.*((G2>weight_real1(1)).*normrnd(miu_real1(1),sigma_real1(1),M,N)+(G2<weight_real1(1)).*normrnd(miu_real1(2),sigma_real1(2),M,N))+1*((R2>TH).*10.*randn(M,N)+(R2<TH).*0.5.*randn(M,N));
%     VV3 = 1i.*((G3>weight_real1(1)).*normrnd(miu_real1(1),sigma_real1(1),M,N)+(G3<weight_real1(1)).*normrnd(miu_real1(2),sigma_real1(2),M,N))+1*((R3>TH).*10.*randn(M,N)+(R3<TH).*0.5.*randn(M,N));
% % VV1=VV1-mean(VV1);
% VV2=VV2-mean(VV2);
% VV3=VV3-mean(VV3);




%     St1 = (0.5+0.5i)*A1*s + VV1; %完整的接收到的信号（滤波，多路阵列汇总前）
%     St2 = (0.5+0.5i)*A2*s + VV2; %完整的接收到的信号（滤波，多路阵列汇总前）
%     St3 = (0.5+0.5i)*A3*s +VV3; %完整的接收到的信号（滤波，多路阵列汇总前）
    VV1 = randn(M,N).*1i+1.*((R1>=TH).*20.*randn(M,N)+(R1<TH).*0.5.*randn(M,N));
    VV2 =randn(M,N).*1i+1.*((R2>=TH).*20.*randn(M,N)+(R2<TH).*0.5.*randn(M,N));
    VV3 = randn(M,N).*1i+1.*((R3>=TH).*20.*randn(M,N)+(R3<TH).*0.5.*randn(M,N));

    St1 = A1*s+VV1 ; %完整的接收到的信号（滤波，多路阵列汇总前）
    St2 = A2*s+VV2; %完整的接收到的信号（滤波，多路阵列汇总前）
    St3 = A3*s+VV3 ; %完整的接收到的信号（滤波，多路阵列汇总前）

    St=[St1 St2 St3];
    % LMS 算法 开始 进行自适应滤波

    d1 = s(b1,:);  % 选取需要辨识的信号，即你需要观察的信号（隐含假设，即需要辨识的信号是什么样的是已知的，在通信中，这是合理的）

    u1 = 0.0005;
   lamda_RLS=0.985; 
   lamda_RMCL=0.995; 
    mu_MCC=0.0005;
    w1 = zeros(M,N*3);  %% 初始化权重向量，一般是0向量
    w_MCC1 = zeros(M,N*3);  
    w_RLS1 = zeros(M,N*3);
    w_RMCL1 = zeros(M,N*3);
    %% 
   Pn = eye(M)*1;
    Pn_RMCL = eye(M)*1;
    for i = 1:N
        Err_LMS(kk,i)=(w1(:,i)-A1(:,b1)/M)'*(w1(:,i)-A1(:,b1)/M);
        Err_MCC(kk,i)=(w_MCC1(:,i) -A1(:,b1)/M)'*(w_MCC1(:,i)  -A1(:,b1)/M);
        Err_RLS(kk,i)=(w_RLS1(:,i) -A1(:,b1)/M)'*(w_RLS1(:,i)-  A1(:,b1)/M);
        Err_RMCL(kk,i)=(w_RMCL1(:,i) -A1(:,b1)/M)'*(w_RMCL1(:,i)-  A1(:,b1)/M);

        y1(i) = w1(:,i)'*St(:,i);
        e1(i) = d1(i) - y1(i);

        y1_MCC(i) = w_MCC1(:,i)'*St(:,i);
        e_MCC1(i) = d1(i) - y1_MCC(i);

  y1_RLS(i) = w_RLS1(:,i)'*St(:,i);
        e_RLS1(i) = d1(i) - y1_RLS(i);

  y1_RMCL(i) = w_RMCL1(:,i)'*St(:,i);
        e_RMCL1(i) = d1(i) - y1_RMCL(i);

        w1(:,i+1) = w1(:,i) + u1 * St(:,i) * conj(e1(i));
        w_MCC1(:,i+1)  =  w_MCC1(:,i)  + mu_MCC * exp(- (abs( e_MCC1(i)))^2/(2*sigma_MCC^2))*conj(  e_MCC1(i))*St(:,i);

    kn = Pn * St(:,i) / ( lamda_RLS+ St(:,i)' * Pn * St(:,i) );
    Pn = 1/lamda_RLS * ( Pn - kn *St(:,i)' * Pn);
     w_RLS1(:,i+1) = w_RLS1(:,i) +kn * conj(e_RLS1(i));


      for k=1:orders
                 P_Rand(i,k)=exp(-1*abs(e_RMCL1(i)-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
            end
            for k=1:orders
                V_Rand(i,k)=WEIGHT_Rand(k)*P_Rand(i,k)/(WEIGHT_Rand*P_Rand(i,:)');
            end
         Q1=0;
         Q2=0;
            for k=1:orders
                Q1=Q1+V_Rand(i,k)/SIGMA_Rand(k)^2;
                  Q2=Q2+V_Rand(i,k)*conj(MIU_Rand(k))/SIGMA_Rand(k)^2;
            end
    kn_RMCL = Pn_RMCL * St(:,i) / ( lamda_RMCL+ Q1*St(:,i)' * Pn_RMCL * St(:,i) );
    Pn_RMCL = 1/lamda_RMCL * ( Pn_RMCL -  Q1*kn_RMCL *St(:,i)' * Pn_RMCL);
     w_RMCL1(:,i+1) = w_RMCL1(:,i) +kn_RMCL * (Q1*conj( e_RMCL1(i))-Q2);
    end

    w_target1=w1(:,i);
    w_target1_MCC1=w_MCC1(:,i) ;
w_target1_RLS1=w_RLS1(:,i) ;
w_target1_RMCL1=w_RMCL1(:,i) ;

    for i = N+1:N*2
        Err_LMS(kk,i)=(w1(:,i)-A2(:,b1)/M)'*(w1(:,i)-A2(:,b1)/M);
        Err_MCC(kk,i)=(w_MCC1(:,i)  -A2(:,b1)/M)'*(w_MCC1(:,i)  -A2(:,b1)/M);
        Err_RLS(kk,i)=(w_RLS1(:,i)  -A2(:,b1)/M)'*(w_RLS1(:,i)  -A2(:,b1)/M);
Err_RMCL(kk,i)=(w_RMCL1(:,i)  -A2(:,b1)/M)'*(w_RMCL1(:,i)  -A2(:,b1)/M);

        y1(i) = w1(:,i)'*St(:,i);
        e1(i) = d1(i-N) - y1(i);

        y1_MCC(i) = w_MCC1(:,i)'*St(:,i);
        e_MCC1(i) = d1(i-N) - y1_MCC(i);

         y1_RLS(i) = w_RLS1(:,i)'*St(:,i);
        e_RLS1(i) = d1(i-N) - y1_RLS(i);

  y1_RMCL(i) = w_RMCL1(:,i)'*St(:,i);
        e_RMCL1(i) = d1(i-N) - y1_RMCL(i);

        w1(:,i+1) = w1(:,i) + u1 * St(:,i) * conj(e1(i));
        w_MCC1(:,i+1)  =  w_MCC1(:,i)  + mu_MCC * exp(- (abs( e_MCC1(i)))^2/(2*sigma_MCC^2))*conj(  e_MCC1(i))*St(:,i);

  kn = Pn * St(:,i) / ( lamda_RLS+ St(:,i)' * Pn * St(:,i) );
    Pn = 1/lamda_RLS * ( Pn - kn *St(:,i)' * Pn);
     w_RLS1(:,i+1) = w_RLS1(:,i) +kn * conj(e_RLS1(i));

  for k=1:orders
                 P_Rand(i,k)=exp(-1*abs(e_RMCL1(i)-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
            end
            for k=1:orders
                V_Rand(i,k)=WEIGHT_Rand(k)*P_Rand(i,k)/(WEIGHT_Rand*P_Rand(i,:)');
            end
         Q1=0;
         Q2=0;
            for k=1:orders
                Q1=Q1+V_Rand(i,k)/SIGMA_Rand(k)^2;
                  Q2=Q2+V_Rand(i,k)*conj(MIU_Rand(k))/SIGMA_Rand(k)^2;
            end
    kn_RMCL = Pn_RMCL * St(:,i) / ( lamda_RMCL+ Q1*St(:,i)' * Pn_RMCL * St(:,i) );
    Pn_RMCL = 1/lamda_RMCL * ( Pn_RMCL -  Q1*kn_RMCL *St(:,i)' * Pn_RMCL);
     w_RMCL1(:,i+1) = w_RMCL1(:,i) +kn_RMCL * (Q1*conj( e_RMCL1(i))-Q2);

    end

    w_target1_1=w1(:,i);
    w_target1_MCC1_1=w_MCC1(:,i) ;
w_target1_RLS1_1=w_RLS1(:,i) ;
w_target1_RMCL1_1=w_RMCL1(:,i) ;

    for i = 2*N+1:N*3
        Err_LMS(kk,i)=(w1(:,i)-A3(:,b1)/M)'*(w1(:,i)-A3(:,b1)/M);
        Err_MCC(kk,i)=(w_MCC1(:,i)  -A3(:,b1)/M)'*(w_MCC1(:,i) -A3(:,b1)/M);
        Err_RLS(kk,i)=(w_RLS1(:,i)  -A3(:,b1)/M)'*(w_RLS1(:,i) -A3(:,b1)/M);
Err_RMCL(kk,i)=(w_RMCL1(:,i)  -A3(:,b1)/M)'*(w_RMCL1(:,i) -A3(:,b1)/M);
        y1(i) = w1(:,i)'*St(:,i);
        e1(i) = d1(i-N*2) - y1(i);

        y1_MCC(i) = w_MCC1(:,i)'*St(:,i);
        e_MCC1(i) = d1(i-N*2) - y1_MCC(i);

        y1_RLS(i) = w_RLS1(:,i)'*St(:,i);
        e_RLS1(i) = d1(i-N*2) - y1_RLS(i);


  y1_RMCL(i) = w_RMCL1(:,i)'*St(:,i);
        e_RMCL1(i) = d1(i-N*2) - y1_RMCL(i);

        w1(:,i+1) = w1(:,i) + u1 * St(:,i) * conj(e1(i));
        w_MCC1(:,i+1)  =  w_MCC1(:,i) + mu_MCC * exp(- (abs( e_MCC1(i)))^2/(2*sigma_MCC^2))*conj(  e_MCC1(i))*St(:,i);

  kn = Pn * St(:,i) / ( lamda_RLS+ St(:,i)' * Pn * St(:,i) );
    Pn = 1/lamda_RLS * ( Pn - kn *St(:,i)' * Pn);
     w_RLS1(:,i+1) = w_RLS1(:,i) +kn * conj(e_RLS1(i));


for k=1:orders
                 P_Rand(i,k)=exp(-1*abs(e_RMCL1(i)-MIU_Rand(k))^2/(SIGMA_Rand(k)^2))/(pi*SIGMA_Rand(k)^2);
            end
            for k=1:orders
                V_Rand(i,k)=WEIGHT_Rand(k)*P_Rand(i,k)/(WEIGHT_Rand*P_Rand(i,:)');
            end
         Q1=0;
         Q2=0;
            for k=1:orders
                Q1=Q1+V_Rand(i,k)/SIGMA_Rand(k)^2;
                  Q2=Q2+V_Rand(i,k)*conj(MIU_Rand(k))/SIGMA_Rand(k)^2;
            end
    kn_RMCL = Pn_RMCL * St(:,i) / ( lamda_RMCL+ Q1*St(:,i)' * Pn_RMCL * St(:,i) );
    Pn_RMCL = 1/lamda_RMCL * ( Pn_RMCL -  Q1*kn_RMCL *St(:,i)' * Pn_RMCL);
     w_RMCL1(:,i+1) = w_RMCL1(:,i) +kn_RMCL * (Q1*conj( e_RMCL1(i))-Q2);

    end



    scan_theta = -89:90;

    for i=1:N*3
        for ii=1:length(scan_theta)
            v = exp(-1j * 2 * pi *d* (0:M-1)'.*sind(scan_theta(ii))/lambda);
            DOA1(ii) = abs(w_MCC1(:,i)'*v);
            DOA3(ii) = abs(w1(:,i)'*v);
            DOA5(ii) = abs(w_RLS1(:,i)'*v);
DOA7(ii) = abs(w_RMCL1(:,i)'*v);
        end
        [r1,index1]=max(DOA1);
        [r3,index3]=max(DOA3);
[r5,index5]=max(DOA5);
[r7,index7]=max(DOA7);
        S1(kk,i)=index1-90;
        S3(kk,i)=index3-90;
S5(kk,i)=index5-90;
S7(kk,i)=index7-90;
    end




    disp(kk);


end


scan_theta = -89:90;
beam1 = zeros(1,length(scan_theta));
beam2 = zeros(1,length(scan_theta));
for i = 1 :length(scan_theta)
    % 构造该方向的方向向量
    v = exp(-1i * 2 * pi *d* (0:M-1)'.*sind(scan_theta(i))/lambda);
    beam_target1(i) = abs(w_target1'*v);
    beam_target1_1(i) = abs(w_target1_1'*v);
    beam1(i) = abs(w1(:,end)'*v);




    beam_MCC1(i) = abs(w_target1_MCC1'*v);
    beam_MCC3(i) = abs(w_target1_MCC1_1'*v);
    beam_MCC5(i) = abs(w_MCC1(:,end)'*v);

     beam_RLS1(i) = abs(w_target1_RLS1'*v);
    beam_RLS3(i) = abs(w_target1_RLS1_1'*v);
    beam_RLS5(i) = abs(w_RLS1(:,end)'*v);

     beam_RMCL1(i) = abs(w_target1_RMCL1'*v);
    beam_RMCL3(i) = abs(w_target1_RMCL1_1'*v);
    beam_RMCL5(i) = abs(w_RMCL1(:,end)'*v);

end


figure(1),subplot(311)
plot(scan_theta,20*log10(beam_target1/max(beam_target1)))
title('Beam Pattern 1(CLMS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);

subplot(312)
plot(scan_theta,20*log10(beam_target1_1/max(beam_target1_1)))
title('Beam Pattern 2(CLMS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);


subplot(313)
plot(scan_theta,20*log10(beam1/max(beam1)))
title('Beam Pattern 3(CLMS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);








figure(2),subplot(311)
plot(scan_theta,20*log10(beam_MCC1/max(beam_MCC1)))
title('Beam Pattern 1(MCCC)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);

subplot(312)
plot(scan_theta,20*log10(beam_MCC3/max(beam_MCC3)))
title('Beam Pattern 2(MCCC)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);

subplot(313)
plot(scan_theta,20*log10(beam_MCC5/max(beam_MCC5)))
title('Beam Pattern 3(MCCC)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);




figure(3),subplot(311)
plot(scan_theta,20*log10(beam_RLS1/max(beam_RLS1)))
title('Beam Pattern 1(CRLS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);

subplot(312)
plot(scan_theta,20*log10(beam_RLS3/max(beam_RLS3)))
title('Beam Pattern 2(CRLS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);

subplot(313)
plot(scan_theta,20*log10(beam_RLS5/max(beam_RLS5)))
title('Beam Pattern 3(CRLS)');
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
box on;grid on;
xlim([-90,90]);



figure(4),subplot(311)
plot(scan_theta,20*log10(beam_RMCL1/max(beam_RMCL1)))
title('Beam Pattern 1(RMCL)');
box on;grid on;
xlim([-90,90]);
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
subplot(312)
plot(scan_theta,20*log10(beam_RMCL3/max(beam_RMCL3)))
title('Beam Pattern 2(RMCL)');
box on;grid on;
xlim([-90,90]);
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');
subplot(313)
plot(scan_theta,20*log10(beam_RMCL5/max(beam_RMCL5)))
title('Beam Pattern 3(RMCL)');
box on;grid on;
xlim([-90,90]);
xlabel('DOA(degree)');
ylabel('Beam Response(dB)');





figure(5),hold on
plot(10*log10(mean(Err_LMS)),'-g','linewidth',1.5);
plot(10*log10(mean(Err_MCC)),'-c','linewidth',1.5);
plot(10*log10(mean(Err_RLS)),'-b','linewidth',1.5);
plot(10*log10(mean(Err_RMCL)),'-y','linewidth',2);
legend('CLMS (\eta=0.0005)','MCCC (\eta=0.0005,\sigma=1)','CRLS (\lambda=0.99)','RMCL (K_R=2,K_I=1,\lambda=0.99)');

xlabel('Time Index');
ylabel('MSD(dB)');
box on;grid on;
xlim([0,N*3]);
ylim([-18.7,-14]);


[m1,index_LMS1]=max(beam_target1);
[m2,index_MCC1]=max(beam_MCC1);



[m5,index_LMS1_1]=max(beam_target1_1);
[m6,index_MCC1_1]=max(beam_MCC3);


[m9,index_LMS1_1_1]=max(beam1);
[m10,index_MCC1_1_1]=max(beam_MCC5);




figure(6),hold on
plot(mean(S3),'-g','linewidth',2);

plot(mean(S1),'-c','linewidth',2);
plot(mean(S5),'-b','linewidth',2);
plot(mean(S7),'-y','linewidth',2.5);
legend('CLMS (\eta=0.0005)','MCCC (\eta=0.0005,\sigma=1)','CRLS (\lambda=0.99)','RMCL (K_R=2,K_I=1,\lambda=0.99)');

xlabel('Time Index');
ylabel('DOA(degree)');
box on;grid on;
xlim([0,N*3]);
ylim([-90,90]);
