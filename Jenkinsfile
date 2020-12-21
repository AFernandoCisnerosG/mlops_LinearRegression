pipeline{

    agent any

    stages{
        stage('Paso 1'){
            steps{
                script{
                    echo "Paso 1 del Jenkisfile"
                }
            }
        }
    }

    post{
        
        always{
            sh "echo 'Ha finalizado el Pipeline de Jenkins'"
        }
        success{
            sh "echo 'Con exito!'"
        }
        failure{
            sh "echo 'Con error!'"
        }

    }

}