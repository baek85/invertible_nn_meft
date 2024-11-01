import torch
import time

@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(500):
            # with amp_autocast():
            model(x)
        torch.cuda.synchronize()

        count = 3000
        print("throughput averaged with {} times".format(count))
        tic1 = time.time()
        for i in range(count):
            # with amp_autocast():
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {count * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)