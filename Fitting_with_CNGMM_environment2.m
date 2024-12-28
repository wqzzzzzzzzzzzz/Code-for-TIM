clear all, close all, clc

   L=2000;
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
 
%     x=raylrnd(8,1,L);
%     x=x-mean(x);

    %         x=rand(1,L)*4-2;
%     sigma=2;
%     x=randn(1,L);
    %   y=randn(1,L)*sigma;

    x=rand(1,L)*10-5;
%      p_rand = rand(1, L);
% 
%     for n=1:L
%         if p_rand(n)<weight_real1(1)
%             x(n)=normrnd(miu_real1(1),sigma_real1(1));
%         else
%             x(n)=normrnd(miu_real1(2),sigma_real1(2));
%         end
%     end

% y=randn(1,L);
y=alpha_noise(1.7,0,1,0,1,L);

  %%  
  MIU_Rand=[miu_real1(1)+j*miu_real2(1) miu_real1(1)+j*miu_real2(2)  miu_real1(2)+j*miu_real2(1)  miu_real1(2)+j*miu_real2(2)];
  SIGMA_Rand=[sqrt(sigma_real1(1)^2+sigma_real2(1)^2) sqrt(sigma_real1(1)^2+sigma_real2(2)^2) sqrt(sigma_real1(2)^2+sigma_real2(1)^2) sqrt(sigma_real1(2)^2+sigma_real2(2)^2)];
  WEIGHT_Rand=[weight_real1(1)*weight_real2(1) weight_real1(1)*weight_real2(2) weight_real1(2)*weight_real2(1) weight_real1(2)*weight_real2(2)];

 MIU_Rand1=[miu_real1(1)+j*miu_real2_4(1) miu_real1(1)+j*miu_real2_4(2) miu_real1(1)+j*miu_real2_4(3) miu_real1(1)+j*miu_real2_4(4) miu_real1(2)+j*miu_real2_4(1)  miu_real1(2)+j*miu_real2_4(2) miu_real1(2)+j*miu_real2_4(3)  miu_real1(2)+j*miu_real2_4(4)];
  SIGMA_Rand1=[sqrt(sigma_real1(1)^2+sigma_real2_4(1)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(2)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(3)^2) sqrt(sigma_real1(1)^2+sigma_real2_4(4)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(1)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(2)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(3)^2) sqrt(sigma_real1(2)^2+sigma_real2_4(4)^2)];
  WEIGHT_Rand1=[weight_real1(1)*weight_real2_4(1) weight_real1(1)*weight_real2_4(2) weight_real1(1)*weight_real2_4(3) weight_real1(1)*weight_real2_4(4) weight_real1(2)*weight_real2_4(1) weight_real1(2)*weight_real2_4(2) weight_real1(2)*weight_real2_4(3) weight_real1(2)*weight_real2_4(4)];


MIU_Rand2=[miu_real1_4(1)+j*miu_real2_4(1) miu_real1_4(1)+j*miu_real2_4(2) miu_real1_4(1)+j*miu_real2_4(3) miu_real1_4(1)+j*miu_real2_4(4) miu_real1_4(2)+j*miu_real2_4(1)  miu_real1_4(2)+j*miu_real2_4(2) miu_real1_4(2)+j*miu_real2_4(3)  miu_real1_4(2)+j*miu_real2_4(4)
    miu_real1_4(3)+j*miu_real2_4(1) miu_real1_4(3)+j*miu_real2_4(2) miu_real1_4(3)+j*miu_real2_4(3) miu_real1_4(3)+j*miu_real2_4(4) miu_real1_4(4)+j*miu_real2_4(1) miu_real1_4(4)+j*miu_real2_4(2) miu_real1_4(4)+j*miu_real2_4(3) miu_real1_4(4)+j*miu_real2_4(4)];
  SIGMA_Rand2=[sqrt(sigma_real1_4(1)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(1)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(2)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(3)^2+sigma_real2_4(4)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(1)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(2)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(3)^2) sqrt(sigma_real1_4(4)^2+sigma_real2_4(4)^2)];
  WEIGHT_Rand2=[weight_real1_4(1)*weight_real2_4(1) weight_real1_4(1)*weight_real2_4(2) weight_real1_4(1)*weight_real2_4(3) weight_real1_4(1)*weight_real2_4(4) weight_real1_4(2)*weight_real2_4(1) weight_real1_4(2)*weight_real2_4(2) weight_real1_4(2)*weight_real2_4(3) weight_real1_4(2)*weight_real2_4(4) weight_real1_4(3)*weight_real2_4(1) weight_real1_4(3)*weight_real2_4(2) weight_real1_4(3)*weight_real2_4(3) weight_real1_4(3)*weight_real2_4(4) weight_real1_4(4)*weight_real2_4(1) weight_real1_4(4)*weight_real2_4(2) weight_real1_4(4)*weight_real2_4(3) weight_real1_4(4)*weight_real2_4(4)];

 


    p_rand1= rand(1, L);

    for n=1:L
        if p_rand1(n)<=(weight_real1(1)*weight_real2(1))
            c(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2(1),sigma_real2(1));
        elseif  p_rand1(n)>((weight_real1(1)*weight_real2(1))) && p_rand1(n)<=(weight_real1(1))
            c(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2(2),sigma_real2(2));
        elseif  p_rand1(n)>(weight_real1(1)) && p_rand1(n)<=(1-(weight_real1(2)*(weight_real2(2))))
            c(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2(1),sigma_real2(1));

        else
            c(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2(2),sigma_real2(2));
        end
    end



    p_rand2= rand(1, L);

    for n=1:L
        if p_rand2(n)<=(weight_real1(1)*weight_real2_4(1))
            c1(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand2(n)>((weight_real1(1)*weight_real2_4(1))) && p_rand2(n)<=(weight_real1(1)*weight_real2_4(1)+weight_real1(1)*weight_real2_4(2))
            c1(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand2(n)>(weight_real1(1)*weight_real2_4(1)+weight_real1(1)*weight_real2_4(2)) && p_rand2(n)<=(weight_real1(1)*weight_real2_4(1)+weight_real1(1)*weight_real2_4(2)+weight_real1(1)*weight_real2_4(3))
            c1(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
     elseif  p_rand2(n)>(weight_real1(1)*weight_real2_4(1)+weight_real1(1)*weight_real2_4(2)+weight_real1(1)*weight_real2_4(3)) && p_rand2(n)<=(weight_real1(1))
            c1(n)=normrnd(miu_real1(1),sigma_real1(1))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));


        elseif p_rand2(n)>(weight_real1(1)) && p_rand2(n)<=(weight_real1(1)+weight_real1(2)*weight_real2_4(1))
            c1(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand2(n)>(weight_real1(1)+(weight_real1(2)*weight_real2_4(1))) && p_rand2(n)<=(weight_real1(1)+weight_real1(2)*weight_real2_4(1)+weight_real1(2)*weight_real2_4(2))
            c1(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand2(n)>(weight_real1(1)+weight_real1(2)*weight_real2_4(1)+weight_real1(2)*weight_real2_4(2)) && p_rand2(n)<=(weight_real1(1)+weight_real1(2)*weight_real2_4(1)+weight_real1(2)*weight_real2_4(2)+weight_real1(2)*weight_real2_4(3))
            c1(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
%      elseif  p_rand2(n)>(weight_real1(1)+weight_real1(2)*weight_real2_4(1)+weight_real1(2)*weight_real2_4(2)+weight_real1(2)*weight_real2_4(3)) && p_rand2(n)<=(weight_real1(1))
%             c(n)=normrnd(miu_real1(4),sigma_real1(4))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));
%             
        else
            c1(n)=normrnd(miu_real1(2),sigma_real1(2))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));
        end
    end



    p_rand3= rand(1, L);

    for n=1:L
        if p_rand3(n)<=(weight_real1_4(1)*weight_real2_4(1))
            c2(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand3(n)>((weight_real1_4(1)*weight_real2_4(1))) && p_rand3(n)<=(weight_real1_4(1)*weight_real2_4(1)+weight_real1_4(1)*weight_real2_4(2))
            c2(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand3(n)>(weight_real1_4(1)*weight_real2_4(1)+weight_real1_4(1)*weight_real2_4(2)) && p_rand3(n)<=(weight_real1_4(1)*weight_real2_4(1)+weight_real1_4(1)*weight_real2_4(2)+weight_real1_4(1)*weight_real2_4(3))
            c2(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
     elseif  p_rand3(n)>(weight_real1_4(1)*weight_real2_4(1)+weight_real1_4(1)*weight_real2_4(2)+weight_real1_4(1)*weight_real2_4(3)) && p_rand3(n)<=(weight_real1_4(1))
            c2(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));


        elseif p_rand3(n)>(weight_real1_4(1)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)*weight_real2_4(1))
            c2(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand3(n)>(weight_real1_4(1)+(weight_real1_4(2)*weight_real2_4(1))) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)*weight_real2_4(1)+weight_real1_4(2)*weight_real2_4(2))
            c2(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)*weight_real2_4(1)+weight_real1_4(2)*weight_real2_4(2)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)*weight_real2_4(1)+weight_real1_4(2)*weight_real2_4(2)+weight_real1_4(2)*weight_real2_4(3))
            c2(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
        elseif p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)*weight_real2_4(1)+weight_real1_4(2)*weight_real2_4(2)+weight_real1_4(2)*weight_real2_4(3))&& p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2))
            c2(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));


    elseif p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)*weight_real2_4(1))
            c2(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+(weight_real1_4(3)*weight_real2_4(1))) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)*weight_real2_4(1)+weight_real1_4(3)*weight_real2_4(2))
            c2(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)*weight_real2_4(1)+weight_real1_4(3)*weight_real2_4(2)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)*weight_real2_4(1)+weight_real1_4(3)*weight_real2_4(2)+weight_real1_4(3)*weight_real2_4(3))
            c2(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
        elseif p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)*weight_real2_4(1)+weight_real1_4(3)*weight_real2_4(2)+weight_real1_4(3)*weight_real2_4(3)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)++weight_real1_4(3))
            c2(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));

    elseif p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4)*weight_real2_4(1))
            c2(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2_4(1),sigma_real2_4(1));
        elseif  p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+(weight_real1_4(4)*weight_real2_4(1))) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4)*weight_real2_4(1)+weight_real1_4(4)*weight_real2_4(2))
            c2(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2_4(2),sigma_real2_4(2));
        elseif  p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4)*weight_real2_4(1)+weight_real1_4(4)*weight_real2_4(2)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4)*weight_real2_4(1)+weight_real1_4(4)*weight_real2_4(2)+weight_real1_4(4)*weight_real2_4(3))
            c2(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2_4(3),sigma_real2_4(3));
        elseif p_rand3(n)>(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4)*weight_real2_4(1)+weight_real1_4(4)*weight_real2_4(2)+weight_real1_4(4)*weight_real2_4(3)) && p_rand3(n)<=(weight_real1_4(1)+weight_real1_4(2)+weight_real1_4(3)+weight_real1_4(4))
            c2(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2_4(4),sigma_real2_4(4));



        end
    end


figure
grid on;
box on;
hold on;
xlabel('Real part');ylabel('Imaginary part');

scatter(x,y,5,'r');

k = boundary(x',y');
plot(x(k),y(k),'r--','LineWidth',2)
% xx=[];
% yy=[];
% 
% xx=[x(k) x(k(1))];
% yy=[y(k) x(k(1))];
% 
% 
% 
% % 傅里叶变换
% XX = fft(xx);
% YY = fft(yy);
% 
% % 设置傅里叶变换的高频分量为0进行平滑处理
% threshold = 6; % 保留前threshold个低频分量
% XX(threshold:end-threshold) = 0;
% YY(threshold:end-threshold) = 0;
% 
% % 傅里叶逆变换得到平滑后的曲线
% x_smooth = ifft(XX, 'symmetric');
% y_smooth = ifft(YY, 'symmetric');
% 
% % 绘制原始和平滑后的曲线
% plot(x_smooth, y_smooth, 'r--', 'LineWidth', 2);


a=real(c);
b=imag(c);
scatter(a,b,5,'b');

kk = boundary(a',b');
% plot(a(kk),b(kk),'c--','LineWidth',2)

aa=[];
bb=[];

aa=[a(kk) a(kk(1))];
bb=[b(kk) b(kk(1))];
% 傅里叶变换
AA= fft(aa);
BB = fft(bb);

% 设置傅里叶变换的高频分量为0进行平滑处理
threshold = 4; % 保留前threshold个低频分量
AA(threshold:end-threshold) = 0;
BB(threshold:end-threshold) = 0;

% 傅里叶逆变换得到平滑后的曲线
a_smooth = ifft(AA, 'symmetric');
b_smooth = ifft(BB, 'symmetric');

% 绘制原始和平滑后的曲线
plot(a_smooth, b_smooth, 'b--', 'LineWidth', 2);
% 
% axis equal;




a1=real(c1);
b1=imag(c1);
scatter(a1,b1,5,'g');

kk1 = boundary(a1',b1');
% plot(a1(kk),b1(kk),'m--','LineWidth',2)

aa1=[];
bb1=[];

aa1=[a1(kk1) a1(kk1(1))];
bb1=[b1(kk1) b1(kk1(1))];
% 傅里叶变换
AA1= fft(aa1);
BB1 = fft(bb1);

% 设置傅里叶变换的高频分量为0进行平滑处理
threshold = 5; % 保留前threshold个低频分量
AA1(threshold:end-threshold) = 0;
BB1(threshold:end-threshold) = 0;

% 傅里叶逆变换得到平滑后的曲线
a_smooth1 = ifft(AA1, 'symmetric');
b_smooth1 = ifft(BB1, 'symmetric');

% 绘制原始和平滑后的曲线
plot(a_smooth1, b_smooth1, 'g--', 'LineWidth', 2);

% axis equal;

a2=real(c2);
b2=imag(c2);
scatter(a2,b2,5,'c');

kk2 = boundary(a2',b2');
% plot(a1(kk),b1(kk),'m--','LineWidth',2)

aa2=[];
bb2=[];

aa2=[a2(kk2) a2(kk2(1))];
bb2=[b2(kk2) b2(kk2(1))];
% 傅里叶变换
AA2= fft(aa2);
BB2 = fft(bb2);

% 设置傅里叶变换的高频分量为0进行平滑处理
threshold = 5; % 保留前threshold个低频分量
AA2(threshold:end-threshold) = 0;
BB2(threshold:end-threshold) = 0;

% 傅里叶逆变换得到平滑后的曲线
a_smooth2 = ifft(AA2, 'symmetric');
b_smooth2 = ifft(BB2, 'symmetric');

% 绘制原始和平滑后的曲线
plot(a_smooth2, b_smooth2, 'c--', 'LineWidth', 2);

legend('Intricate Environment 2','Noise Profile','CNGMM 1(K_R=2,K_I=2)','Profile of CNGMM 1','CNGMM 2(K_R=2,K_I=4)','Profile of CNGMM 2','CNGMM 3(K_R=4,K_I=4)','Profile of CNGMM 3)','NumColumns',2);
% legend('Intricate Environment 2','CGMM(K_R=2,K_I=2)','CGMM(K_R=2,K_I=4)','CGMM(K_R=4,K_I=4)',' Fitting Noise Profile with CGMM(K_R=4,K_I=4)')
% 
xlim([-8,8]);
ylim([-80,130]);

