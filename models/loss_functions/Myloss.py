import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k

class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        self.kernel_left_data = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        self.kernel_right_data = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        self.kernel_up_data = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        self.kernel_down_data = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        
        self.pool = nn.AvgPool2d(4)
    
    def create_kernels(self, device):
        kernel_left = torch.FloatTensor(self.kernel_left_data).to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(self.kernel_right_data).to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(self.kernel_up_data).to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(self.kernel_down_data).to(device).unsqueeze(0).unsqueeze(0)
        return kernel_left, kernel_right, kernel_up, kernel_down
    
    def forward(self, org, enhance):
        b, c, h, w = org.shape
        device = org.device 

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        kernel_left, kernel_right, kernel_up, kernel_down = self.create_kernels(device)

        weight_diff = torch.max(
            torch.tensor([1.0], device=device) + 
            10000 * torch.min(org_pool - torch.tensor([0.3], device=device), 
                             torch.tensor([0.0], device=device)),
            torch.tensor([0.5], device=device)
        )
        
        E_1 = torch.mul(torch.sign(enhance_pool - torch.tensor([0.5], device=device)), 
                       enhance_pool - org_pool)

        D_org_left = F.conv2d(org_pool, kernel_left, padding=1)
        D_org_right = F.conv2d(org_pool, kernel_right, padding=1)
        D_org_up = F.conv2d(org_pool, kernel_up, padding=1)
        D_org_down = F.conv2d(org_pool, kernel_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, kernel_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, kernel_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, kernel_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, kernel_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        
        E = (D_left + D_right + D_up + D_down)

        return E



class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 4, 7]):
        super().__init__()
        self.layers = layers
        self.vgg = torchvision.models.vgg19(pretrained=True).features[:max(layers)+1].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, student_img, teacher_img):
        loss = 0
        for i, layer in enumerate(self.vgg):
            student_img = layer(student_img)
            teacher_img = layer(teacher_img)
            if i+1 in self.layers:
                loss += F.mse_loss(student_img, teacher_img.detach())
        return loss

class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
    def forward(self, x ):
        b,c,h,w = x.shape
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        

        k = torch.mean(k)
        return k



class MultiRegionExposureLoss(nn.Module):
    def __init__(self, patch_size=8, 
                 dark_target=1.2,      # Very aggressive for dark regions
                 mid_target=0.35,       # Strong boost for mid-tones
                 bright_target_factor=0.9,  # Slightly suppress bright regions
                 dark_threshold=0.2,   # More aggressive dark region definition
                 mid_threshold=0.45,
                 bright_transition_width=0.6,     # Between thresholds = mid-tones
                 bright_threshold=0.75):
        super(MultiRegionExposureLoss, self).__init__()
        self.patch_size = patch_size
        self.pool = nn.AvgPool2d(patch_size)
        
        # Region-specific targets
        self.dark_target = dark_target
        self.mid_target = mid_target
        self.bright_target_factor = bright_target_factor
        
        self.dark_threshold = dark_threshold
        self.mid_threshold = mid_threshold
        self.bright_threshold = bright_threshold
        
        # Soft transition parameters
        self.dark_transition_width = 0.05
        self.mid_transition_width = 0.05
        self.bright_transition_width = 0.08

    def create_soft_masks(self, orig_mean_down):
        # Dark mask: high value for very dark regions
        dark_mask = torch.sigmoid((self.dark_threshold - orig_mean_down) / self.dark_transition_width)
        
        # Mid-tone mask: high value for mid-brightness regions
        mid_mask_lower = torch.sigmoid((orig_mean_down - self.dark_threshold) / self.dark_transition_width)
        mid_mask_upper = torch.sigmoid((self.mid_threshold - orig_mean_down) / self.mid_transition_width)
        mid_mask = mid_mask_lower * mid_mask_upper
        
        # Bright mask: high value for very bright regions
        bright_mask = torch.sigmoid((orig_mean_down - self.bright_threshold) / self.bright_transition_width)
        
        # Normalize masks to sum to 1
        total_mask = dark_mask + mid_mask + bright_mask + 1e-8
        dark_mask = dark_mask / total_mask
        mid_mask = mid_mask / total_mask
        bright_mask = bright_mask / total_mask
        
        return dark_mask, mid_mask, bright_mask

    def compute_target_brightness(self, orig_mean_down):
        # Compute target brightness with region-specific values
        dark_mask, mid_mask, bright_mask = self.create_soft_masks(orig_mean_down)
        
        # Dark regions target fixed value
        dark_target = torch.ones_like(orig_mean_down) * self.dark_target
        
        # Mid regions target fixed value
        mid_target = torch.ones_like(orig_mean_down) * self.mid_target
        
        # Bright regions target scaled original value
        bright_target = orig_mean_down * self.bright_target_factor
        
        return (dark_target * dark_mask + 
                mid_target * mid_mask + 
                bright_target * bright_mask)

    def forward(self, x, original_x):
        b, c, h, w = x.shape
        
        # Compute mean intensity of the original image
        orig_mean = torch.mean(original_x, dim=1, keepdim=True)
        orig_mean_down = self.pool(orig_mean)
        
        # Compute target brightness
        target_brightness = self.compute_target_brightness(orig_mean_down)
        
        # Compute mean intensity of enhanced image
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean_down = self.pool(x_mean)
        
        # Calculate region-specific exposure loss
        region_loss = (x_mean_down - target_brightness) ** 2
        
        #Preserve local contrast
        orig_mean_pooled = F.interpolate(orig_mean, size=x_mean.shape[2:], mode='bilinear', align_corners=False)
        x_mean_pooled = F.interpolate(x_mean_down, size=x_mean.shape[2:], mode='bilinear', align_corners=False)
        
        orig_contrast = torch.abs(orig_mean - orig_mean_pooled)
        enhanced_contrast = torch.abs(x - x_mean_pooled)
        
        contrast_loss = torch.mean(torch.abs(enhanced_contrast - orig_contrast))
        
        return torch.mean(region_loss) + 0.2 * contrast_loss
    
