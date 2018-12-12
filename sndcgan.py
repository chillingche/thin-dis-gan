import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as tvutils
from torchvision import datasets, transforms
import thindidata
import thindiarch as arch
from sndiscriminator import Discriminator
from sngenerator import Generator
import utils
import config

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=config.ROOT, help="path to dataset")
parser.add_argument(
    "--batch-size", type=int, default=128, help="input batch size")
parser.add_argument(
    "--nz", type=int, default=128, help="size of latent z vector")
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
    "--ckpt-d", default=config.CKPT, help="directory to checkpoint")
parser.add_argument(
    "--eval-d", default=config.EVAL, help="directory to output")
opt = parser.parse_args()
try:
    os.makedirs(opt.ckpt_d)
    os.makedirs(opt.eval_d)
except OSError:
    pass

cudnn.benchmark = True
en_cuda = opt.cuda and torch.cuda.is_available()
dataloader = thindidata.get_dataloader(
    thindidata.Cifar10Data, opt.root, "all", opt.batch_size, num_workers=2)
device = torch.device("cuda" if en_cuda else "cpu")
if opt.ngpu == 1:
    torch.cuda.set_device(1)
ngpu = opt.ngpu
netD = Discriminator(ngpu).to(device)
netG = Generator(ngpu).to(device)

if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)

optimD = optim.Adam(netD.parameters(), lr=4 * opt.lr, betas=(opt.beta1, 0.999))
optimG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    nround = len(dataloader) // 10
    for i, data in enumerate(dataloader, 0):
        image, label = data
        real = image.to(device)
        batch_size = real.size(0)
        z = torch.randn(batch_size, opt.nz, 1, 1, device=device)

        optimD.zero_grad()
        output = netD(real)
        errD_real = arch.HingeAdvLoss.get_d_real_loss(output)
        D_x = output.mean().item()

        fake = netG(z)
        output = netD(fake.detach())
        errD_fake = arch.HingeAdvLoss.get_d_fake_loss(output)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        errD.backward()
        optimD.step()

        optimG.zero_grad()
        output = netD(fake)
        errG = arch.HingeAdvLoss.get_g_loss(output)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()

        if i % nround == 0:
            print(
                '%s [%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (utils.current_str_time(), epoch, opt.niter, i,
                   len(dataloader), errD.item(), errG.item(), D_x, D_G_z1,
                   D_G_z2))
    utils.save_image(real, '%s/real_samples.png' % opt.eval_d, normalize=True)
    fake = netG(fixed_noise)
    utils.save_image(
        fake.detach(),
        '%s/fake_samples_epoch_%03d.png' % (opt.eval_d, epoch),
        normalize=True)
    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.ckpt_d, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.ckpt_d, epoch))
