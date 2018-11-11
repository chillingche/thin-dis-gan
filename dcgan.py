import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as tvutils
import thindidata
import archmodule as arch
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="/data/cifar", help="path to dataset")
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
parser.add_argument("--ckpt-d", default="../output.d/thindi-gan/ckpt.d", help="directory to checkpoint")
parser.add_argument("--eval-d", default="../output.d/thindi-gan/eval.d", help="directory to output")
opt = parser.parse_args()
try:
    os.makedirs(opt.ckpt_d)
    os.makedirs(opt.eval_d)
except OSError:
    pass

cudnn.benchmark = True
en_cuda = opt.cuda and torch.cuda.is_available()
dataloader = thindidata.get_dataloader(thindidata.Cifar10Data, opt.root, "all",
                                       opt.batch_size)
device = torch.device("cuda" if en_cuda else "cpu")
ngpu = opt.ngpu
netD = arch.Discriminator(ngpu).to(device)
netG = arch.Generator(ngpu).to(device)
netD.apply(utils.weight_init)
netG.apply(utils.weight_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))
criterion = nn.MSELoss()
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        image, label = data
        netD.zero_grad()
        real = image.to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size, ), real_label, device=device)
        output = netD(real)

        # errD_real = 0.5*criterion(output, label)
        errD_real = arch.HingleAdvLoss.get_d_real_loss(output)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        # errD_fake = 0.5*criterion(output, label)
        errD_fake = arch.HingleAdvLoss.get_d_fake_loss(output)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        # errG = 0.5*criterion(output, label)
        errG = arch.HingleAdvLoss.get_g_loss(output)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()

        print(
            '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(),
               D_x, D_G_z1, D_G_z2))
    tvutils.save_image(
        real, '%s/real_samples.png' % opt.eval_d, normalize=True)
    fake = netG(fixed_noise)
    tvutils.save_image(
        fake.detach(),
        '%s/fake_samples_epoch_%03d.png' % (opt.eval_d, epoch),
        normalize=True)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.ckpt_d, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.ckpt_d, epoch))
