import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
import thindidata
from sngenerator import SKetchGenerator, PhotoGenerator
from sndiscriminator import SketchDiscriminator, PhotoDiscriminator
from thindiarch import HingeAdvLoss
import utils
import utils.vision_util as tvutils
import config

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=config.ROOT, help="path to dataset")
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
    thindidata.ThindiCifar10Data,
    opt.root,
    "train",
    opt.batch_size,
    num_workers=8)

device = torch.device("cuda" if en_cuda else "cpu")
if opt.ngpu == 1 and int(torch.cuda.device_count()) > 1:
    torch.cuda.set_device(1)

ngpu = opt.ngpu
netSD = SketchDiscriminator(ngpu).to(device)
netPD = PhotoDiscriminator(ngpu).to(device)
netSG = SKetchGenerator(ngpu).to(device)
netPG = PhotoGenerator(ngpu).to(device)
netSD.apply(utils.weight_init)
netSG.apply(utils.weight_init)
netPD.apply(utils.weight_init)
netPG.apply(utils.weight_init)
if opt.netG != "":
    netSG.load_state_dict(torch.load(opt.netG))
if opt.netD != "":
    netSD.load_state_dict(torch.load(opt.netD))
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimSD = optim.Adam(
    netSD.parameters(), lr=4 * opt.lr, betas=(opt.beta1, 0.999))
optimSG = optim.Adam(netSG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimPD = optim.Adam(
    netPD.parameters(), lr=4 * opt.lr, betas=(opt.beta1, 0.999))
params = list()
for param in netSG.parameters():
    params.append(param)
for param in netPG.parameters():
    params.append(param)
optimPG = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

# lr decay
schedulerSD = ExponentialLR(optimSD, gamma=0.99)
schedulerSG = ExponentialLR(optimSG, gamma=0.99)
schedulerPD = ExponentialLR(optimPD, gamma=0.99)
schedulerPG = ExponentialLR(optimPG, gamma=0.99)

log_interval = 100
batch_iter = iter(dataloader)

for i in range(opt.niter):
    if utils.is_new_epoch_began(i, dataloader):
        batch_iter = iter(dataloader)
        schedulerSD.step()
        schedulerSG.step()
        schedulerPD.step()
        schedulerPG.step()
    image, sketch, label = next(batch_iter)
    netSD.zero_grad()
    real_sk = sketch.to(device)
    batch_size = real_sk.size(0)
    output = netSD(real_sk)
    errSD_real = HingeAdvLoss.get_d_real_loss(output)
    SD_x = output.mean().item()

    netPD.zero_grad()
    real_ph = image.to(device)
    output = netPD(real_ph)
    errPD_real = HingeAdvLoss.get_d_real_loss(output)
    PD_x = output.mean().item()

    noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
    fake_sk = netSG(noise)
    output = netSD(fake_sk.detach())
    errSD_fake = HingeAdvLoss.get_d_fake_loss(output)
    errSD = errSD_real + errSD_fake
    errSD.backward()
    optimSD.step()

    fake_ph = netPG(fake_sk, noise)
    output = netPD(fake_ph.detach())
    errPD_fake = HingeAdvLoss.get_d_fake_loss(output)
    errPD = errPD_real + errPD_fake
    errPD.backward()
    optimPD.step()

    netSG.zero_grad()
    output = netSD(fake_sk)
    errSG = HingeAdvLoss.get_g_loss(output)
    errSG.backward(retain_graph=True)
    optimSG.step()
    SD_z = output.mean().item()

    netPG.zero_grad()
    output = netPD(fake_ph)
    errPG = HingeAdvLoss.get_g_loss(output)
    errPG.backward()
    optimPG.step()
    PD_z = output.mean().item()

    if i % log_interval == 0:
        print(
            '%s [%d/%d] errSD: %.4f errSG: %.4f errPD: %.4f errPG: %.4f SD(x): %.4f SD(SG(z)): %.4f PD(x): %.4f PD(PG(z)): %.4f'
            % (utils.current_str_time(), i, opt.niter, errSD.item(),
               errSG.item(), errPD.item(), errPG.item(), SD_x, SD_z, PD_x,
               PD_z))
        with torch.no_grad():
            fake_sk = netSG(fixed_noise)
            fake = netPG(fake_sk, fixed_noise)
            browserwriter.add_scalars(
                "OUT/LOSS", {
                    "errSD": errSD.item(),
                    "errSG": errSG.item(),
                    "errPD": errPD.item(),
                    "errPG": errPG.item()
                }, i)
            browserwriter.add_scalars("OUT/D_OUT", {
                "SD_x": SD_x,
                "SD_z": SD_z,
                "PD_x": PD_x,
                "PD_z": PD_z
            }, i)
            browserwriter.add_image(
                "G_OUT/fake_sk",
                tvutils.make_grid(fake_sk.detach(), normalize=False), i)
            browserwriter.add_image(
                "G_OUT/fake_ph",
                tvutils.make_grid(fake.detach(), normalize=True), i)
browserwriter.close()
# torch.save(netSG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.ckpt_d, epoch))
# torch.save(netSD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.ckpt_d, epoch))
