6.6 最终完善界面

python ./start.py

6.2 全面修改界面+文件名称的确定版本备份

python ./start.py # CUDA_VISIBLE_DEVICES="0,1,2,3"

CUDA_VISIBLE_DEVICES="0,1,2,3" python ./retriever.py

CUDA_VISIBLE_DEVICES="0,1,2,3" python ./retriever_judge.py

6.1 再回到174服务器重新配置law-retriever环境(发现系统驱动太老只能用低版本torch2.1.0) 6.2 demo均运行成功 (13600MB / 13200MB) ：以下成功命令

CUDA_VISIBLE_DEVICES="0,1,2,3" python ./demo_r.py (或CUDA_VISIBLE_DEVICES="0"也足够)
CUDA_VISIBLE_DEVICES="0,1,2,3" python ./demo.py (或CUDA_VISIBLE_DEVICES="0"也是用0123)

-------------------------------------------------------------------------------------------------

C:/Users/LENOVO/.conda/envs/qh/python.exe g:/Law-Retriever/src/demo_r.py # 必须进文件夹src才能使用'./retriver/law_embs.pkl'

C:/Users/LENOVO/.conda/envs/qh/python.exe g:/Law-Retriever/src/demo.py # 也需src 否则model无法访问

5.31 首次运行本地显卡 

C:/Users/LENOVO/.conda/envs/qh/python.exe g:/Law-Retriever/src/demo_r.py

直接本地3060显卡运行demo_r测试检索效果-- (必须指定python为qh的python 方法是右键运行run python file in terminal 否则用的是默认python版本而不是qh安装的虚拟环境的python版本 因为本地的终端不是anaconda、无法切换虚拟环境

python ./demo_r.py
python ./demo.py

CUDA_VISIBLE_DEVICES="0" python ./demo_r.py
CUDA_VISIBLE_DEVICES="0" python ./demo.py

