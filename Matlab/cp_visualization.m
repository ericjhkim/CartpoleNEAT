% Space-Static Visualization

clear;
close all;

load cp_data.mat

% Data variables
x = x_list;
dx = dx_list;
theta = -theta_list+pi;
dtheta = dtheta_list;

dt = t_list(2)-t_list(1);

% Cart
c_w = 0.25;
c_h = 0.1;

% Pole
p_w = 0.04;
p_h = 0.5;

% Canvas
figure(1);clf;

% Ground
yline(0,'Color',[0.5 0.5 0.5],'LineWidth',2,'alpha',1);
hold on;

cart = polyshape([0 0 c_w c_w],[0 c_h c_h 0]);
pole = polyshape([0 0 p_w p_w],[0 p_h p_h 0]);
set(gcf,'position',get(0,'ScreenSize')); % Fill screensize

txt = text(x_list(1)-0.14,0-0.1,'0');
txt.FontSize = 30;

for t = 1:length(t_list)

    axis([-10 10 -1 2]);
    daspect([1 1 1]);

    % Move cartpole
    cart_i = translate(cart,[x(t)-c_w/2 0]);
    pole_i = rotate(pole,rad2deg(theta(t)+pi),[p_w/2 0]);
%     pole_i = rotate(pole,rad2deg(-theta(t)),[p_w/2 0]);
    pole_i = translate(pole_i,[(x(t)-p_w/2) c_h]);

    % Plot cartpole
    c = plot(cart_i,'FaceColor','k','FaceAlpha',1);
    p = plot(pole_i,'FaceColor','r','FaceAlpha',1);

    % Hinge
    h = plot(x(t), c_h, '.k', 'MarkerSize',30);
    
    % Text
    delete(txt);
    txt = text(x_list(t)-0.14,0-0.1,string(t_list(t)));
    txt.FontSize = 30;

    pause(dt); % Set framerate
    
    if t < length(t_list)
        delete(c);
        delete(p);
        delete(h);
    end
    
end

% % Canvas settings
% axis([-10 1 -0.25 1]);
% daspect([1 1 1]);
