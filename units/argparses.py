import argparse

def file_path_get():
    parser = argparse.ArgumentParser(
        prog='myprogram',          # 程序名（默认为 sys.argv[0]）
        description='获取文件路径',      # 程序功能描述
        epilog='python main.py "path/path"',           # 帮助信息结尾的文本
        add_help=True             # 是否自动添加 -h/--help 选项
    )

    # 位置参数（必需参数）
    parser.add_argument(
        'input_file',              # 参数名
        type=str,                  # 参数类型
        help='输入文件路径'          # 帮助信息
    )

    args = parser.parse_args()

    print(f"文件地址 {args.input_file}")
    return args.input_file

if __name__ == "__main__":
    file_path_get()