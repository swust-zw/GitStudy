#ecoding=gbk
from flask import Flask,render_template,request,jsonify,url_for
from app_ocr import base64toimg,circle_ocr,ocr_baidu
import binascii
app = Flask(__name__)
@app.route('/enterprise-seals/ocr', methods = ["POST"])
def success():
    flag=0
    datas = request.get_json(force=True)
    res_list = []
    res_num = [i for i in range(len(datas['base64']))]
    for data in datas['base64']:
        imgdata = data
        code = 10260000
        res = ""
        type = ''
        try:
            base64toimg.img(imgdata)
            flag = 1
            file = 'img\yingzhang.png'
            base64toimg.init_img(file)
            flag = 2
            judge = base64toimg.judgle(file)
            res_file = 'img/result.png'
            if judge == 0:
                type='circle'
                circle_ocr.split(file)
                res = ocr_baidu.circle(res_file)
            elif judge == 1:
                type='rect'
                ocr_baidu.rect_init(file)
                res = ocr_baidu.rect(res_file)
            message = '识别成功'
            if res=='' or type=='':
                code=10260002
                message = '识别错误'
        except:
            if flag==0:
                code = 10260001
                message = '输入错误'
            elif flag==1:
                code = 10260003
                message = '图片错误'
            else:
                code = 10260004
                message = '其他错误'
        return_data = {
            'code':code,
            "message":message,
            "result":{
                'sealName': res,
                'type': type,
            }
        }
        res_list.append(return_data)
    res_dict=dict(zip(res_num,res_list))
    return res_dict
@app.route('/login/',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      imgdata = request.form['nm']
      base64toimg.img(imgdata)
      file = 'img\yingzhang.png'
      judge=base64toimg.judgle(file)
      res_file = 'img/result.png'
      if judge==0:
          circle_ocr.split(file)
          res=ocr_baidu.circle(res_file)
      elif judge==1:
          # res='矩形'
          ocr_baidu.rect_init(file)
          res=ocr_baidu.rect(res_file)
      return render_template('index.html',result='结果:'+res)
   else:
      user = request.args.get('nm')
      return render_template('index.html',result='结果:')
@app.route('/test/',methods=['POST'])
def test():
    # try:
    # data = request.json
    data = request.get_json(force=True)
    d='shaha'
    print("测试数据",data["student_name"],d)

    # data = request.get_json()
    if data["student_name"] == "Jack":
        return_data = {"name": "Jack", "age": 19, "Gender": "Male"}
    elif data["student_name"] == "Rose":
        return_data = {"name": "Rose", "age": 18, "Gender": "Female"}
    else:
        return_data = "No student info named " + str(data["student_name"])
    return jsonify(return_data)
    # except:
    #     return "error"

if __name__ == '__main__':
    app.run(debug=True,port =8080, )
