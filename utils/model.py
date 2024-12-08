import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import *
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel



class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
class MMIUNet_GuideDecoder(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(MMIUNet_GuideDecoder, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]    # 224*224
        feature_dim = [768,384,192,96]

        channels = [96, 192, 384, 768] # ConvNeXt-T
        
        self.fusion1 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
        self.fusion2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
        self.fusion3 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
        self.fusion4 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        encoder_feats = []
        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        vis_feat, lan_feat = self.fusion1(image_features[1], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion2(image_features[2], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion3(image_features[3], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion4(image_features[4], text_embeds[-1])
        encoder_feats.append(vis_feat)

        if len(encoder_feats[0].shape) == 4: 
            # encoder_feats = encoder_feats[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            encoder_feats = [rearrange(item,'b c h w -> b (h w) c') for item in encoder_feats] 

        os32 = encoder_feats[3]
        os16 = self.decoder16(os32, encoder_feats[2], text_embeds[-1])
        os8 = self.decoder8(os16, encoder_feats[1], text_embeds[-1])
        os4 = self.decoder4(os8, encoder_feats[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out

class MMIUNet_V2(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(MMIUNet_V2, self).__init__()

        # self.encoder = VisionModel(vision_type, project_dim)
        in_chans = 3
        depths = [3,3,9,3]
        dims = [96,192,384,768]
        drop_path_rate = 0.
        layer_scale_init_value = 1e-6
        head_init_scale = 1.

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], 1)

        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]    # 224*224
        feature_dim = [768, 384, 192, 96]
        channels = [96, 192, 384, 768] # ConvNeXt-T

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion1 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
        self.fusion2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
        self.fusion3 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
        self.fusion4 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)
        
        self.decode4 = Decoder(channels[3],channels[2])
        self.decode3 = Decoder(channels[2],channels[1])
        self.decode2 = Decoder(channels[1],channels[0])

        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        encoder_feats = []
        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w', c=3)

        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, _ = text_output['feature'],text_output['project']
        txt = text_embeds[-1]

        x = self.downsample_layers[0](image)
        x = self.stages[0](x)
        res = x
        
        vis_feat, txt_feat = self.fusion1(x, txt)
        encoder_feats.append(vis_feat)
        x = self.downsample_layers[1](vis_feat + res)
        x = self.stages[1](x)
        res = x
        
        vis_feat, txt_feat = self.fusion2(x, txt + txt_feat)
        encoder_feats.append(vis_feat)
        x = self.downsample_layers[2](vis_feat + res)
        x = self.stages[2](x)
        res = x
        
        vis_feat, txt_feat = self.fusion3(x, txt + txt_feat)    
        encoder_feats.append(vis_feat)
        x = self.downsample_layers[3](vis_feat + res)
        x = self.stages[3](x)

        vis_feat, txt_feat = self.fusion4(x, txt + txt_feat)    
        encoder_feats.append(vis_feat)

        d4 = self.decode4(encoder_feats[3], encoder_feats[2])
        d3 = self.decode3(d4, encoder_feats[1])
        d2 = self.decode2(d3, encoder_feats[0])
        os1 = self.decoder1(d2)
        out = self.out(os1).sigmoid()

        return out
