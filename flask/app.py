from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
import logging
import pymysql
import os
from werkzeug.utils import secure_filename
from patch_trainer import PatchTrainer  # 导入你的PatchTrainer类

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查数据库连接
def check_database():
    try:
        # 尝试连接数据库
        connection = pymysql.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
            port=Config.MYSQL_PORT
        )
        
        logger.info("数据库连接成功！")
        
        # 创建表
        with connection.cursor() as cursor:
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE 'users'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                # 创建用户表
                sql = """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    password VARCHAR(200) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cursor.execute(sql)
                connection.commit()
                logger.info("users 表创建成功！")
            else:
                logger.info("users 表已存在！")
                
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        return False

# 初始化Flask应用
app = Flask(__name__)
app.config.from_object(Config)

# 在应用初始化之前添加
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 检查数据库连接
if not check_database():
    logger.error("无法连接到数据库，应用将退出！")
    exit(1)

# 初始化数据库
db = SQLAlchemy(app)

# 用户模型
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<User {self.username}>'

# 配置上传文件的目录
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/register', methods=['POST'])
def register():
    """用户注册接口"""
    try:
        data = request.get_json()
        
        # 验证请求数据
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({
                'code': 400,
                'message': '请提供用户名和密码'
            }), 400
            
        username = data['username']
        password = data['password']
        
        # 验证用户名和密码的长度
        if len(username) < 3 or len(password) < 6:
            return jsonify({
                'code': 400,
                'message': '用户名长度至少3个字符，密码长度至少6个字符'
            }), 400
            
        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            return jsonify({
                'code': 400,
                'message': '用户名已存在'
            }), 400
            
        # 创建新用户
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'code': 200,
            'message': '注册成功'
        })
        
    except Exception as e:
        logger.error(f"注册失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': '服务器内部错误'
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """用户登录接口"""
    try:
        data = request.get_json()
        
        # 验证请求数据
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({
                'code': 400,
                'message': '请提供用户名和密码'
            }), 400
            
        username = data['username']
        password = data['password']
        
        # 查找用户
        user = User.query.filter_by(username=username).first()
        
        # 验证用户名和密码
        if not user or not check_password_hash(user.password, password):
            return jsonify({
                'code': 401,
                'message': '用户名或密码错误'
            }), 401
            
        return jsonify({
            'code': 200,
            'message': '登录成功',
            'data': {
                'user_id': user.id,
                'username': user.username
            }
        })
        
    except Exception as e:
        logger.error(f"登录失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': '服务器内部错误'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'code': 200,
        'message': 'OK'
    })

@app.route('/api/generate_patch', methods=['POST'])
def generate_patch():
    """生成对抗补丁的接口"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({
                'code': 400,
                'message': '没有上传文件'
            }), 400
            
        file = request.files['image']
        target_class = request.form.get('target_class', type=int)
        
        if file.filename == '':
            return jsonify({
                'code': 400,
                'message': '没有选择文件'
            }), 400
            
        if not target_class and target_class != 0:
            return jsonify({
                'code': 400,
                'message': '请提供目标类别'
            }), 400
            
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 初始化训练器
            trainer = PatchTrainer(target_class)
            
            # 预处理图像
            img_tensor = trainer.preprocess_image(filepath)
            
            # 训练生成补丁
            patch_path, patched_image_path = trainer.train(img_tensor)
            
            # 返回结果
            return jsonify({
                'code': 200,
                'message': '生成成功',
                'data': {
                    'patch_path': patch_path,
                    'patched_image_path': patched_image_path
                }
            })
            
    except Exception as e:
        logger.error(f"生成补丁失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'服务器内部错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
