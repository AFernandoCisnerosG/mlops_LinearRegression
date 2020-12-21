pipeline{

    agent any

    stages{
        stage('Paso 1'){
            steps{
                script{
                    sh "Paso 1 del Jenkisfile"
                }
            }
        }
    }

    notifications{
        always{
            sh echo "Finalizo el Pipeline de Jenkins"
        }
        success{
            mail(to:fernando.cisnerosgaytan@gmail.com, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Bien, funcinó!")
        }
        failure{
            mail(to:fernando.cisnerosgaytan@gmail.com, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Boo, falló!")
        }
        unstable{
            mail(to:fernando.cisnerosgaytan@gmail.com, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Huh, es inestable!")
        }
    }

}