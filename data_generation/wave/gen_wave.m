clc; clear all; close all;

%2D WAVE EQUATION u_tt = c^2(uxx+uyy)
%with initial condition  u(x,y,0) = sin(p*pi*x)*sin(q*pi*y), 0<x<1 0<y<1
% and boundary conditions u(0,y,t) = u(1,y,t)= u(x,0,t)= u(x,1,t) = 0 t>0

c = 4;
dx = 4/127;  %128*128 grid
dy = dx;
sigma = 1; gamma = 1/sqrt(2); %Courant-Friedrich Stability Condition
dt = sigma*(dx/c);
%dt = 1/119
t = 0:dt:1; x = 0:dx:4; y = 0:dy:4;
u = zeros(length(x),length(y),length(t));
p = 2; q = 1;


u(:,:,1) = transpose(sin(p.*pi.*x))*sin(q.*pi.*y); %u(x,y,0) = sin(p*pi*x)*sin(q*pi*y)

%Analytic solution
SOL = zeros(length(x),length(y),length(t));
S = transpose(sin(p.*pi.*x))*sin(q.*pi.*y);

for i=1:length(t)
    SOL(:,:,i) = S*cos(c.*pi.*(sqrt(p^2+q^2).*t(i)));
end

data = SOL;
num_ic = 30;

for i=1:num_ic-1
    p = 3 * rand;
    q = 2 * rand;
    SOL = zeros(length(x),length(y),length(t));
    S = transpose(sin(p.*pi.*x))*sin(q.*pi.*y);

    for i=1:length(t)
        SOL(:,:,i) = S*cos(c.*pi.*(sqrt(p^2+q^2).*t(i)));
    end

    data = cat(4,data,SOL);
end

data = permute(data,[4,1,2,3]);
save('./wave_2d.mat', 'data')



