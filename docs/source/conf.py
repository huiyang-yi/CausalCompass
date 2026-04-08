import os
import sys

# 1. 指向源代码路径
sys.path.insert(0, os.path.abspath('../..'))

# 2. Mock (模拟) 重型依赖
autodoc_mock_imports = [
    'torch', 'jax', 'jaxlib', 'jaxtyping', 'equinox', 'optax', 'chex', 'tensorboard',
    'numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'networkx', 'matplotlib', 'igraph',
    'tigramite', 'lingam', 'benchmark_mi', 'dcor',
    'tqdm', 'omegaconf', 'einops', 
    'causalnex', 'graphviz', 'pygraphviz', 'pydot' # <--- graphviz 全家桶
]

# 3. 项目信息
project = 'CausalCompass'
copyright = '2026, CausalCompass Team'
author = 'Xiaojian Shen, Huiyang Yi'
release = '0.1.0'

# 4. 加载插件
extensions = [
    'sphinx.ext.autodoc',      # 自动提取代码注释
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格注释
    'sphinx.ext.viewcode',     # 添加 [source] 源码链接按钮
    'sphinx.ext.autosummary'
]

autosummary_generate = True
autodoc_preserve_defaults = True
# 5. 主题设置
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# ==================================================
# NEW: LOGO 和 静态文件配置
# ==================================================

# 告诉 Sphinx 静态资源放在哪里 (必须是 list)
html_static_path = ['_static']

# 设置 Logo (路径是相对于 _static 的，实际文件在 source/_static/logo.png)
# 请确保您的图片文件名和这里一致！
html_logo = '_static/logo.png' 
# html_logo = '_static/logo.svg'

# 设置浏览器标签页小图标 (可选)
# html_favicon = '_static/favicon.ico'

# 主题显示选项
html_theme_options = {
    'logo_only': True,       # 如果设为 True，则只显示 Logo，不显示 "CausalCompass" 文字标题
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9', # 可选：修改左侧导航栏背景色
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# ==================================================

# 6. 支持的文件扩展名
source_suffix = {
    '.rst': 'restructuredtext',
}