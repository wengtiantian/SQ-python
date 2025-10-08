sq-python/
├── apps/                               # 各业务模块目录
│   ├── sq/                             # 服务质量评价sq模块
│   │   ├── curd/                       # 数据增删改查逻辑层（CRUD）
│   │   │   ├── __init__.py
│   │   │   └── curd_sq.py              # sq模块的数据库操作接口
│   │   ├── models/                     # 数据模型定义层（ORM模型）
│   │   │   ├── __init__.py
│   │   │   └── sq.py                   # sq模块的数据库模型定义
│   │   ├── schemas/                    # 数据验证与序列化层（Pydantic）
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── all_dim.txt                 # 全维度配置/词表文件
│   │   ├── my_dim_dict.txt             # 自定义维度字典
│   │   ├── PMI_sent_dict2.json         # PMI语义词典，用于情感或语义分析
│   │   └── views.py                    # 接口路由与业务逻辑控制器
│   └── __init__.py
│
├── common/                             # 通用工具与中间层
│   ├── __init__.py
│   ├── curd_base.py                    # 通用CRUD基类，提供基本数据库操作模板
│   ├── deps.py                         # 依赖注入模块，用于FastAPI依赖声明
│   ├── error_code.py                   # 全局错误码定义
│   ├── exceptions.py                   # 自定义异常类
│   ├── middleware.py                   # 全局中间件定义（如日志、跨域、认证）
│   ├── resp.py                         # 标准化响应结构
│   └── security.py                     # 鉴权与加密安全相关方法
│
├── configs/                            # 配置文件目录
│   ├── .env                            # 环境变量配置文件
│   ├── logging_config.conf             # 日志配置文件
│   └── supervisor.conf.example         # Supervisor进程管理示例配置
│
├── core/                               # 核心模块配置
│   ├── __init__.py
│   ├── config.py                       # 全局配置加载逻辑
│   ├── constants.py                    # 常量定义
│   └── logger.py                       # 日志系统初始化
│
├── db/                                 # 数据库连接与基础层
│   ├── __init__.py
│   ├── base_class.py                   # ORM基类定义
│   ├── cache.py                        # 缓存逻辑（如Redis操作）
│   ├── mongo.py                        # MongoDB数据库操作模块
│   └── session.py                      # 数据库会话管理（SQLAlchemy）
│
├── log/                                # 日志文件目录
│   ├── error.log                       # 错误日志
│   └── info.log                        # 运行信息日志
│
├── media/                              # 输出文件存储目录
│   ├── Bars/                           # 柱状图输出
│   ├── Pareto_chart/                   # 帕累托图输出
│   ├── Radar/                          # 雷达图输出
│   └── WordCloud/                      # 词云图输出
│
├── utils/                              # 工具类模块
│   ├── __init__.py
│   ├── captcha_code.py                 # 验证码生成工具
│   ├── email.py                        # 邮件发送工具
│   ├── encrypt.py                      # 加密与解密工具
│   ├── loggers.py                      # 日志封装工具
│   └── transform.py                    # 通用数据转换函数
│
├── workers/                            # 异步任务队列模块（Celery）
│   ├── __init__.py
│   ├── celery_tasks.py                 # 任务定义文件
│   └── celeryconfig.py                 # Celery配置文件
│
├── alice_color.png                     # 项目Logo或展示图片
├── dir_tree.py                         # 项目目录结构生成脚本
├── main.py                             # 项目入口文件（FastAPI启动文件）
├── nohup.out                           # 后台运行日志输出文件
├── requirements.txt                    # Python依赖包列表
└── simhei.ttf                          # 中文字体文件（用于生成图表中文字显示）
