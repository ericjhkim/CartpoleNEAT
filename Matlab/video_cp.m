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

% Visualization
save = 1;
if save == 1
    gifname = 'C:\EK_Projects\CP_NEAT\Matlab\balancing.gif';
elseif save == 2
    v = VideoWriter(['C:\EK_Projects\CP_NEAT\Matlab\balancing']);
    v.FrameRate = 1/dt;
    open(v);
end

% Cart
c_w = 0.25;
c_h = 0.1;
cart = polyshape([0 0 c_w c_w],[0 c_h c_h 0]);

% Pole
p_w = 0.04;
p_h = 0.5;
pole = polyshape([0 0 p_w p_w],[0 p_h p_h 0]);

for t = 1:length(t_list)
    fig = figure(1); clf;
    
    % Ground
    yline(0,'Color',[0.5 0.5 0.5],'LineWidth',2,'alpha',1);
    hold on;

    set(gcf,'position',get(0,'ScreenSize')); % Fill screensize

    % Stopwatch
    txt = text(x_list(1)-0.14,0-0.1,'0');
    txt.FontSize = 30;
    
    % Canvas
    axis([-3 3 -1 2]);
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
    
    frame = getframe(fig); 
    
    % Write to the GIF File
    if save == 1
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if t == 1
            imwrite(imind,cm,gifname,'gif','DelayTime',dt,'Loopcount',inf);
        elseif t == length(t_list)
            imwrite(imind,cm,gifname,'gif','DelayTime',3,'WriteMode','append');
        else
            imwrite(imind,cm,gifname,'gif','DelayTime',dt,'WriteMode','append');
        end
    elseif save == 2
        writeVideo(v,frame);
    end
end

if save == 2
    close(v);
end
