import pymysql
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet import Facenet
from PIL import Image
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",  # 数据库用户名
    "password": "",  # 数据库密码
    "db": "",  # 数据库名称
}

# Flask 应用初始化
app = Flask(__name__)
CORS(app, resources=r"/*")

# 加载 Facenet 模型
model = Facenet()


def get_db_connection():
    """获取数据库连接"""
    try:
        db = pymysql.connect(**DB_CONFIG)
        return db
    except pymysql.Error as e:
        logging.error(f"数据库连接失败: {e}")
        raise


def close_db_connection(db):
    """关闭数据库连接"""
    if db:
        db.close()


def validate_image_path(image_path):
    """验证图像路径是否存在"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    return image_path


@app.route("/face", methods=["POST"])
def face_recognition():
    """人脸识别接口"""
    result = []
    info = {"code": 500, "msg": "服务器异常"}  # 默认返回错误信息
    db = None

    try:
        # 获取请求参数
        username = request.form.get("username")
        nowimage_path = request.form.get("nowimage")

        if not username or not nowimage_path:
            info["code"] = 400
            info["msg"] = "参数缺失"
            return jsonify([info])

        # 验证图像路径
        nowimage_path = validate_image_path(nowimage_path)

        # 连接数据库
        db = get_db_connection()
        cursor = db.cursor()

        # 查询数据库
        query = "SELECT id, username, userimage FROM student WHERE username = %s"
        cursor.execute(query, (username,))
        data = cursor.fetchone()

        if not data:
            info["code"] = 404
            info["msg"] = "用户未找到"
            return jsonify([info])

        # 解析查询结果
        user_id, user_name, user_image_path = data
        user_image_path = validate_image_path(user_image_path)

        # 打开图像
        user_image = Image.open(user_image_path)
        now_image = Image.open(nowimage_path)

        # 人脸识别
        probability = model.detect_image(user_image, now_image)

        # 返回结果
        if probability[0] < 0.9:
            info["code"] = 200
            info["msg"] = "Same Sample"
        else:
            info["code"] = 200
            info["msg"] = "Different Sample"

        result.append(info.copy())
        return jsonify(result)

    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        info["msg"] = "图像文件不存在"
        return jsonify([info])
    except pymysql.Error as e:
        logging.error(f"数据库错误: {e}")
        info["msg"] = "数据库操作失败"
        return jsonify([info])
    except Exception as e:
        logging.error(f"未知错误: {e}")
        return jsonify([info])
    finally:
        # 关闭数据库连接
        close_db_connection(db)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8899)
    except Exception as e:
        logging.error(f"Flask 应用启动失败: {e}")
    finally:
        logging.info("应用已关闭")
