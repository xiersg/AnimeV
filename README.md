# AnimeV
***将现实中人类的动作实时映射到二次元人物中***

## 实现流程

1. 通过opencv获取摄像头信息
2. 通过MediaPipe模型实现人体节点检查
3. 将各个关键点映射到体块上
4. 通过Pygame渲染体块

## 项目结构
```
主文件夹:
    processed_character_parts/character_parts   缩小版的体块图片
    try3.py     实现粗略映射，初步验证项目可行性
    my_v.py     在try3基础上提升模块化
    
    untis:
        argprses.py     增加命令行参数
        img_tf          图片处理器

```