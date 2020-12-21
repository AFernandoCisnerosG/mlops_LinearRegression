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
            echo "Finalizo el Pipeline de Jenkins"
        }
        success{
            mail(to:fernando.cisnerosgaytan, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Bien, funcinó!")
        }
        failure{
            mail(to:fernando.cisnerosgaytan, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Boo, falló!")
        }
        unstable{
            mail(to:fernando.cisnerosgaytan, subject:"SUCCESS:${currentBuild.fullDisplayName}",
                body: "Huh, es inestable!")
        }
    }

}