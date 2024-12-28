clear all, close all, clc
L=2000;
%% Mean
miu_real1=[0 0];
miu_real2=[0 0];
miu_real1_4=[0 0 0 0];
%% Standard 
sigma_real1=[1.6 20];
sigma_real2=[1 1];
sigma_real1_4=[0.5 1.7 4.8 33.3];
%% Weight
weight_real1=[0.98 0.02];
weight_real2=[0.5 0.5];
weight_real1_4=[0.36 0.56 0.07 0.01];

%%
MIU_Rand=[miu_real1(1)+j*miu_real2(1) miu_real1(1)+j*miu_real2(2)  miu_real1(2)+j*miu_real2(1)  miu_real1(2)+j*miu_real2(2)];
SIGMA_Rand=[sqrt(sigma_real1(1)^2+sigma_real2(1)^2) sqrt(sigma_real1(1)^2+sigma_real2(2)^2) sqrt(sigma_real1(2)^2+sigma_real2(1)^2) sqrt(sigma_real1(2)^2+sigma_real2(2)^2)];
WEIGHT_Rand=[weight_real1(1)*weight_real2(1) weight_real1(1)*weight_real2(2) weight_real1(2)*weight_real2(1) weight_real1(2)*weight_real2(2)];

MIU_Rand1=zeros(1,8)+j*zeros(1,8);
SIGMA_Rand1=[sqrt(sigma_real1_4(1)^2+sigma_real2(1)^2) sqrt(sigma_real1_4(1)^2+sigma_real2(2)^2) sqrt(sigma_real1_4(2)^2+sigma_real2(1)^2) sqrt(sigma_real1_4(1)^2+sigma_real2(2)^2) sqrt(sigma_real1_4(3)^2+sigma_real2(1)^2) sqrt(sigma_real1_4(3)^2+sigma_real2(2)^2) sqrt(sigma_real1_4(4)^2+sigma_real2(1)^2) sqrt(sigma_real1_4(4)^2+sigma_real2(2)^2)];
WEIGHT_Rand1=[weight_real1_4(1)*weight_real2(1) weight_real1_4(1)*weight_real2(2) weight_real1_4(2)*weight_real2(1) weight_real1_4(2)*weight_real2(2) weight_real1_4(3)*weight_real2(1) weight_real1_4(3)*weight_real2(2) weight_real1_4(4)*weight_real2(1) weight_real1_4(4)*weight_real2(2)];


x=alpha_noise(1.7,0,1,0,1,L);
y=randn(1,L);

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
    if p_rand2(n)<=(weight_real1_4(1)*weight_real2(1))
        c1(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2(1),sigma_real2(1));
    elseif  p_rand2(n)>((weight_real1_4(1)*weight_real2(1))) && p_rand2(n)<=weight_real1_4(1)
        c1(n)=normrnd(miu_real1_4(1),sigma_real1_4(1))+1j*normrnd(miu_real2(2),sigma_real2(2));
    elseif  p_rand2(n)>weight_real1_4(1) && p_rand2(n)<=(weight_real1_4(1)+weight_real1_4(2)*weight_real2(1))
        c1(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2(1),sigma_real2(1));
    elseif  p_rand2(n)>(weight_real1_4(1)+weight_real1_4(2)*weight_real2(1))  && p_rand2(n)<=(weight_real1_4(1)+weight_real1_4(2))
        c1(n)=normrnd(miu_real1_4(2),sigma_real1_4(2))+1j*normrnd(miu_real2(2),sigma_real2(2));


    elseif p_rand2(n)>(weight_real1_4(1)+weight_real1_4(2)) && p_rand2(n)<=((weight_real1_4(1)+weight_real1_4(2))+weight_real1_4(3)*weight_real2(1))
        c1(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2(1),sigma_real2(1));
    elseif  p_rand2(n)>((weight_real1_4(1)+weight_real1_4(2))+weight_real1_4(3)*weight_real2(1)) && ((weight_real1_4(1)+weight_real1_4(2))+weight_real1_4(3))
        c1(n)=normrnd(miu_real1_4(3),sigma_real1_4(3))+1j*normrnd(miu_real2(2),sigma_real2(2));
    elseif  p_rand2(n)>( ((weight_real1_4(1)+weight_real1_4(2))+weight_real1_4(3))) && p_rand2(n)<=((weight_real1_4(1)+weight_real1_4(2))+weight_real1_4(3))+weight_real1_4(4)*weight_real2(1)
        c1(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2(1),sigma_real2(1));
    else
        c1(n)=normrnd(miu_real1_4(4),sigma_real1_4(4))+1j*normrnd(miu_real2(2),sigma_real2(2));
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


a=real(c);
b=imag(c);
scatter(a,b,5,'b');
kk = boundary(a',b');


aa=[];
bb=[];

aa=[a(kk) a(kk(1))];
bb=[b(kk) b(kk(1))];
% 傅里叶变换
AA= fft(aa);
BB = fft(bb);

% 设置傅里叶变换的高频分量为0进行平滑处理
threshold = 5; % 保留前threshold个低频分量
AA(threshold:end-threshold) = 0;
BB(threshold:end-threshold) = 0;

% 傅里叶逆变换得到平滑后的曲线
a_smooth = ifft(AA, 'symmetric');
b_smooth = ifft(BB, 'symmetric');

% 绘制原始和平滑后的曲线
plot(a_smooth, b_smooth, 'b--', 'LineWidth', 2);
% axis equal;




a1=real(c1);
b1=imag(c1);
scatter(a1,b1,5,'g');

kk1 = boundary(a1',b1');

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
legend('Intricate Environment 1','Noise Profile','CNGMM 1(K_R=2,K_I=1)','Profile of CNGMM 1','CNGMM 2(K_R=4,K_I=1)','Profile of CNGMM 2');
xlim([-40,40]);
ylim([-4,10]);

