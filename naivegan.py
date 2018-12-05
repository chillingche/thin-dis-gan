import os
import os.path
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as tvutils
import thindidata
import thindiarch as arch
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.path.expanduser("~/data/cifar"), help="path to dataset")
parser.add_argument(
    "--batch-size", type=int, default=128, help="input batch size")
parser.add_argument(
    "--nz", type=int, default=100, help="size of latent z vector")
parser.add_argument("--ngf", type=int, default=128, help="width of netG")
parser.add_argument("--ndf", type=int, default=128, help="width of netD")
parser.add_argument(
    "--niter", type=int, default=25, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
parser.add_argument("--cuda", action="store_true", help="enable cuda")
parser.add_argument(
    "--ngpu", type=int, default=2, help="number of gpus to use")
parser.add_argument("--netG", default="", help="path to netG state dict")
parser.add_argument("--netD", default="", help="path to netD state dict")
parser.add_argument(
    "--ckpt-d",
    default=os.path.expanduser("~/.local/output.d/thindi-gan/ckpt.d"),
    help="directory to checkpoint")
parser.add_argument(
    "--eval-d",
    default=os.path.expanduser("~/.local/output.d/thindi-gan/eval.d"),
    help="directory to output")
opt = parser.parse_args()
try:
    os.makedirs(opt.ckpt_d)
    os.makedirs(opt.eval_d)
except OSError:
    pass

cudnn.benchmark = True
en_cuda = opt.cuda and torch.cuda.is_available()
dataloader = thindidata.get_dataloader(
    thindidata.ThindiCifar10Data,
    opt.root,
    "all",
    opt.batch_size,
    num_workers=8)
device = torch.device("cuda" if en_cuda else "cpu")
ngpu = opt.ngpu
netSD = arch.SketchDiscriminator(ngpu).to(device)
netSG = arch.SKetchGenerator(ngpu).to(device)
netSD.apply(utils.weight_init)
netSG.apply(utils.weight_init)
if opt.netG != "":
    netSG.load_state_dict(torch.load(opt.netG))
if opt.netD != "":
    netSD.load_state_dict(torch.load(opt.netD))
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimSD = optim.Adam(netSD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimSG = optim.Adam(netSG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        image, sketch, label = data
        netSD.zero_grad()
        real_sk = sketch.to(device)
        batch_size = real_sk.size(0)
        output = netSD(real_sk)

        errSD_real = arch.HingeAdvLoss.get_d_real_loss(output)
        errSD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
        fake_sk = netSG(noise)
        output = netSD(fake_sk.detach())
        errSD_fake = arch.HingeAdvLoss.get_d_fake_loss(output)
        errSD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errSD_real + errSD_fake
        optimSD.step()

        netSG.zero_grad()
        output = netSD(fake_sk)
        errSG = arch.HingeAdvLoss.get_g_loss(output)
        errSG.backward()
        D_G_z2 = output.mean().item()
        optimSG.step()

        print(
            '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader), errD.item(), errSG.item(),
               D_x, D_G_z1, D_G_z2))
    tvutils.save_image(
        real_sk, '%s/real_samples.png' % opt.eval_d, normalize=True)
    with torch.no_grad():
        fake_sk = netSG(fixed_noise)
        tvutils.save_image(
            fake_sk.detach(),
            '%s/fake_sk_epoch_%03d.png' % (opt.eval_d, epoch),
            normalize=False)
    # torch.save(netSG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.ckpt_d, epoch))
    # torch.save(netSD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.ckpt_d, epoch))
