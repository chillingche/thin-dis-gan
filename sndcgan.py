import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.utils as tvutils
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
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
    if not (os.path.exists(opt.ckpt_d) and os.path.exists(opt.eval_d)):
        raise OSError("Failed to make ckpt and eval directories")

browserwriter = SummaryWriter(config.TFLOGDIR)
cudnn.benchmark = True
en_cuda = opt.cuda and torch.cuda.is_available()
dataloader = thindidata.get_dataloader(
    thindidata.Cifar10Data, opt.root, "all", opt.batch_size, num_workers=2)
# mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
# std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
# dataloader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         opt.root,
#         train=True,
#         download=True,
#         # transform=transforms.Compose(
#         #     [transforms.ToTensor(),
#         #      transforms.Normalize(mean, std)])
#         transform=transforms.ToTensor()),
#     batch_size=opt.batch_size,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True)
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
# lr decay
schedulerD = ExponentialLR(optimD, gamma=0.99)
schedulerG = ExponentialLR(optimG, gamma=0.99)

log_interval = 100
batch_iter = iter(dataloader)
for i in range(opt.niter):
    if utils.is_new_epoch_began(i, dataloader):
        batch_iter = iter(dataloader)
        schedulerD.step()
        schedulerG.step()
    image, label = next(batch_iter)
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

    if i % log_interval == 0:
        print(
            '%s [%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (utils.current_str_time(), i, opt.niter, errD.item(),
               errG.item(), D_x, D_G_z1, D_G_z2))
        browserwriter.add_scalars("OUT/LOSS", {
            "D_loss": errD.item(),
            "G_loss": errG.item()
        }, i)
        browserwriter.add_scalars("OUT/OUTPUT", {
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2
        }, i)
        fake = netG(fixed_noise)
        utils.save_image(
            fake.detach(),
            '%s/fake_samples_iter_%06d.png' % (opt.eval_d, i),
            normalize=True)
        browserwriter.add_image("FAKE",
                                utils.make_grid(fake.detach(), normalize=False),
                                i)
browserwriter.close()
# torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.ckpt_d, epoch))
# torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.ckpt_d, epoch))
