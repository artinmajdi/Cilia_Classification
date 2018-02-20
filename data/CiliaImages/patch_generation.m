% Code to generat patches from images. Need to change the image name "img1,
% img2, etc.." according to the image that we plan to output the patches
% for. Also need to change the proper directories to store the patches.
% Usage
%
% load img8
% [ct1 l1] = bwlabeln(c1,8);
% [ct2 l2] = bwlabeln(c2,8);
%------------------------------- For Class 1 -----------------------------%
% for i = 1:l1
%     stats1 = regionprops(ct1==i,'Centroid');
%     if(stats1.Centroid(1,2)-16 < 1 || stats1.Centroid(1,2)+15 > 2048 || stats1.Centroid(1,1)-16 < 1 || stats1.Centroid(1,1)+15 > 2048)
%         t = padarray(img8,[16 16],'symmetric');
%         temp = t(stats1.Centroid(1,2)+16-16:stats1.Centroid(1,2)+16+15,stats1.Centroid(1,1)+16-16:stats1.Centroid(1,1)+16+15,:);
%     else 
%     temp = img8(stats1.Centroid(1,2)-16:stats1.Centroid(1,2)+15,stats1.Centroid(1,1)-16:stats1.Centroid(1,1)+15,:);
%     end
%     if(i<10)
%         name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_000' num2str(i) '.tif'];
%         else if(i>=10 && i<100)
%                 name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_00' num2str(i) '.tif'];
%             else
%                 name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_0' num2str(i) '.tif'];
%         end
%     end
%     imwrite(temp,name,'tif');
% end
% %------------------------------- For Class 2 -----------------------------%
% for i = 1:l2
%     stats1 = regionprops(ct2==i,'Centroid');
%     if(stats1.Centroid(1,2)-16 < 1 || stats1.Centroid(1,2)+15 > 2048 || stats1.Centroid(1,1)-16 < 1 || stats1.Centroid(1,1)+15 > 2048)
%         t = padarray(img8,[16 16],'symmetric');
%         temp = t(stats1.Centroid(1,2)+16-16:stats1.Centroid(1,2)+16+15,stats1.Centroid(1,1)+16-16:stats1.Centroid(1,1)+16+15,:);
%     else 
%     temp = img8(stats1.Centroid(1,2)-16:stats1.Centroid(1,2)+15,stats1.Centroid(1,1)-16:stats1.Centroid(1,1)+15,:);
%     end
%     if(i<10)
%         name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_000' num2str(i) '.tif'];
%         else if(i>=10 && i<100)
%                 name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_00' num2str(i) '.tif'];
%             else
%                 name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_0' num2str(i) '.tif'];
%         end
%     end
%     imwrite(temp,name,'tif');
% end
%


[ct1 l1] = bwlabeln(c1,8);
[ct2 l2] = bwlabeln(c2,8);

%------------------------------- For Class 1 -----------------------------%
for i = 1:l1
    stats1 = regionprops(ct1==i,'Centroid');
    if(stats1.Centroid(1,2)-16 < 1 || stats1.Centroid(1,2)+15 > 2048 || stats1.Centroid(1,1)-16 < 1 || stats1.Centroid(1,1)+15 > 2048)
        t = padarray(img8,[16 16],'symmetric');
        temp = t(stats1.Centroid(1,2)+16-16:stats1.Centroid(1,2)+16+15,stats1.Centroid(1,1)+16-16:stats1.Centroid(1,1)+16+15,:);
    else 
    temp = img8(stats1.Centroid(1,2)-16:stats1.Centroid(1,2)+15,stats1.Centroid(1,1)-16:stats1.Centroid(1,1)+15,:);
    end
    if(i<10)
        name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_000' num2str(i) '.tif'];
        else if(i>=10 && i<100)
                name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_00' num2str(i) '.tif'];
            else
                name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class1\img8_0' num2str(i) '.tif'];
        end
    end
    imwrite(temp,name,'tif');
end
%------------------------------- For Class 2 -----------------------------%
for i = 1:l2
    stats1 = regionprops(ct2==i,'Centroid');
    if(stats1.Centroid(1,2)-16 < 1 || stats1.Centroid(1,2)+15 > 2048 || stats1.Centroid(1,1)-16 < 1 || stats1.Centroid(1,1)+15 > 2048)
        t = padarray(img8,[16 16],'symmetric');
        temp = t(stats1.Centroid(1,2)+16-16:stats1.Centroid(1,2)+16+15,stats1.Centroid(1,1)+16-16:stats1.Centroid(1,1)+16+15,:);
    else 
    temp = img8(stats1.Centroid(1,2)-16:stats1.Centroid(1,2)+15,stats1.Centroid(1,1)-16:stats1.Centroid(1,1)+15,:);
    end
    if(i<10)
        name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_000' num2str(i) '.tif'];
        else if(i>=10 && i<100)
                name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_00' num2str(i) '.tif'];
            else
                name = ['C:\Users\Sundaresh Ram\Desktop\SaIL Lab\Cilia Project\Cilia Images\class2\img8_0' num2str(i) '.tif'];
        end
    end
    imwrite(temp,name,'tif');
end
