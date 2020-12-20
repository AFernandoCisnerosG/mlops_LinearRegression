pipeline{

    agent any

    stages{
        stage('Build'){
            steps{
                echo 'Building...'
            }
        }
    }

    notifications{
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