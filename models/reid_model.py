import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

PRETRAINED_MODELS = {
    "resnet18": { "load": lambda : models.resnet18(pretrained=True), "feature_size": 512},
    "resnet50": { "load": lambda : models.resnet50(pretrained=True), "feature_size": 2048}
    # More pretrained models here e.g. alexnet, vgg16, etc
}

class FinetunedModel(nn.Module):
    def __init__(self, architecture, n_classes):
        super(FinetunedModel, self).__init__()
        self.architecture = architecture

        self.backbone = PRETRAINED_MODELS[architecture]["load"]()
        self.feature_size = PRETRAINED_MODELS[architecture]["feature_size"]
        print(f"Backbone feature size: {self.feature_size}")
        self.finetune(self.backbone, n_classes)

    def finetune(self, model, n_classes):
        model_name = model.__class__.__name__
        if model_name.lower().startswith("resnet"):
            self.features = nn.Sequential(*list(model.children())[:-1])
            #we are using output as 1 because we are going to use binary classifcation
            self.age_classifier        =   nn.Sequential(nn.Linear(self.feature_size,1),
                                                         nn.Sigmoid()) #4
            self.backpack_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1),
                                                         nn.Sigmoid())
            self.bag_classifier        =   nn.Sequential(nn.Linear(self.feature_size,1),
                                                         nn.Sigmoid())
            # self.handbag_classifier    =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.clothes_classifier    =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.down_classifier       =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.up_classifier         =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.hair_classifier       =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.hat_classifier        =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.gender_classifier     =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upblack_classifier    =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upwhite_classifier    =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upred_classifier      =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.uppurple_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upyellow_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upgray_classifier     =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upblue_classifier     =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.upgreen_classifier    =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downblack_classifier  =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downwhite_classifier  =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downpink_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downpurple_classifier =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downyellow_classifier =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downgray_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downblue_classifier   =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downgreen_classifier  =   nn.Sequential(nn.Linear(self.feature_size,1))
            # self.downbrown_classifier  =   nn.Sequential(nn.Linear(self.feature_size,1))

    def forward(self, x, get_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if get_features:
            return x
        
        # age                   = self.age_classifier(x)
        backpack_classifier   = self.backpack_classifier(x)
        bag_classifier        = self.bag_classifier(x)  
        # handbag_classifier    = self.handbag_classifier(x)  
        # clothes_classifier    = self.clothes_classifier(x)  
        # down_classifier       = self.down_classifier(x)     
        # up_classifier         = self.up_classifier(x)      
        # hair_classifier       = self.hair_classifier(x)     
        # hat_classifier        = self.hat_classifier(x)      
        # gender_classifier     = self.gender_classifier   
        # upblack_classifier    = self.upblack_classifier(x) 
        # upwhite_classifier    = self.upwhite_classifier(x) 
        # upred_classifier      = self.upred_classifier(x)    
        # uppurple_classifier   = self.uppurple_classifier(x) 
        # upyellow_classifier   = self.upyellow_classifier(x) 
        # upgray_classifier     = self.upgray_classifier(x)   
        # upblue_classifier     = self.upblue_classifier(x)   
        # upgreen_classifier    = self.upgreen_classifier(x)  
        # downblack_classifier  = self.downblack_classifier(x)  
        # downwhite_classifier  = self.downwhite_classifier(x)  
        # downpink_classifier   = self.downpink_classifier(x)  
        # downpurple_classifier = self.downpurple_classifier(x) 
        # downyellow_classifier = self.downyellow_classifier(x) 
        # downgray_classifier   = self.downgray_classifier(x)   
        # downblue_classifier   = self.downblue_classifier(x)   
        # downgreen_classifier  = self.downgreen_classifier(x)  
        # downbrown_classifier  = self.downbrown_classifier(x) 
        return backpack_classifier, bag_classifier 
                #backpack_classifier,
                #bag_classifier
                # handbag_classifier,
                # clothes_classifier,
                # down_classifier,
                # up_classifier,
                # hair_classifier,
                # hat_classifier,
                # gender_classifier,
                # upblack_classifier,
                # upwhite_classifier,
                # upred_classifier,
                # uppurple_classifier,
                # upyellow_classifier,
                # upgray_classifier,
                # upblue_classifier,
                # upgreen_classifier,
                # downblack_classifier,
                # downwhite_classifier,
                # downpink_classifier,
                # downpurple_classifier,
                # downyellow_classifier,
                # downgray_classifier,
                # downblue_classifier,
                # downgreen_classifier,
                # downbrown_classifier
                #]
