from flask import Flask, render_template, request

app = Flask(__name__) 



def suma(a, b):
    for n in (a,b):
        if not isinstance(n, int) and not isinstance(n, float):
            return TypeError
    return a + b


@app.route('/')
def home():
    #return 'Hola'
    return render_template('index.html')

@app.route('/user')
def user():
    print("Yo soy el usuario")
    r = suma(10, 3)
    return 'probando la funcion suma %s' % (r)




if __name__ == '__main__':
    app.run(debug=True) 