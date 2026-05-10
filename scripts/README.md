# 上传脚本说明

## 登录信息

登录指令：ssh -p 35533 root@connect.bjb2.seetacloud.com
密码：KaT71pAMgXsj

## 上传代码

在项目根目录运行：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon
./scripts/deploy_upload.sh
```

脚本会把代码上传到：

```text
root@connect.bjb2.seetacloud.com:/root/ocean-subsurface-recon
```

脚本会排除下面这些大文件或本地文件：

```text
.git
.conda
data
checkpoints
outputs
runs
wandb
logs
__pycache__
*.pyc
*.tar.gz
```

## 上传数据

数据比较大，建议用 `rsync` 单独上传：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon

rsync -avP \
  -e "ssh -p 35533" \
  data/ \
  root@connect.bjb2.seetacloud.com:/root/ocean-subsurface-recon/data/
```

`rsync` 支持断点续传，并且再次运行时只会上传变化过的文件。

## 只上传某一个数据子目录

例如，只上传 `data/raw/mur_sst`：

```bash
rsync -avP \
  -e "ssh -p 35533" \
  data/raw/mur_sst/ \
  root@connect.bjb2.seetacloud.com:/root/ocean-subsurface-recon/data/raw/mur_sst/
```

## 在服务器上解压代码

上传脚本执行完成后，会打印服务器上的解压命令。这个命令会先备份旧项目目录，再解压新代码。

手动执行版本：

```bash
ssh -p 35533 root@connect.bjb2.seetacloud.com

if [ -d /root/ocean-subsurface-recon ]; then
  mv /root/ocean-subsurface-recon /root/ocean-subsurface-recon_backup_$(date +%Y%m%d_%H%M%S)
fi

mkdir -p /root/ocean-subsurface-recon
tar -xzf /root/ocean-subsurface-recon.tar.gz -C /root/ocean-subsurface-recon
```
