class Params:
    def __init__(self):
       
        self.name = 'Pix2Pix'
    
        self.pretrain_path = None
        self.vgg_model = 'pretrain_models/vgg19-dcbb9e9d.pth'
        self.lr = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.99

        self.use_exp = True
        self.lambda_surface = 2.0
        self.lambda_texture = 2.0
        self.lambda_content = 200
        self.lambda_tv = 1e4
       
        self.lambda_exp = 1.0
       

        self.train_src_root = '/StyleTransform/DATA/ffhq-2w/img'
        self.train_tgt_root = '/StyleTransform/DATA/select-style-gan'
        self.val_src_root = '/StyleTransform/DATA/dmloghq-1k/img'
        self.val_tgt_root = '/StyleTransform/DATA/select-style-gan'
        self.score_info = 'pretrain_models/all_express_mean.npy'

        self.infer_batch_size = 2

