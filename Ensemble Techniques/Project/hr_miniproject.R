rm(list=ls())
gc()
setwd("D:/Great Lakes PGPDSE/Great Lakes/13 Ensemble Techniques/Mini Project")
## Reading the data set
hr=read.csv("HR_Employee_Attrition_Data.csv",stringsAsFactors = FALSE)
summary(hr)
# Data set have 2940 observation and 34 variables is there 
str(hr)
summary(hr)
# There is no missing values in the dataset
# Changing value in attrition 0 and 1 format
hr$Attrition[hr$Attrition=="No"]=0
hr$Attrition[hr$Attrition=="Yes"]=1
# Changing value in BusinessTravel 1,2  and 3
hr$BusinessTravel[hr$BusinessTravel=="Non-Travel"]=1
hr$BusinessTravel[hr$BusinessTravel=="Travel_Frequently"]=2
hr$BusinessTravel[hr$BusinessTravel=="Travel_Rarely"]=3
# Changing value in Department 1, 2 and 3
hr$Department[hr$Department=="Human Resources"]=1
hr$Department[hr$Department=="Research & Development"]=2
hr$Department[hr$Department=="Sales"]=3
# Changing value of EducationField into the 1 to 4 format
hr$EducationField[hr$EducationField=="Human Resources"]=1
hr$EducationField[hr$EducationField=="Life Sciences"]=2
hr$EducationField[hr$EducationField=="Markerting"]=3
hr$EducationField[hr$EducationField=="Medical"]=4
hr$EducationField[hr$EducationField=="Other"]=5
hr$EducationField[hr$EducationField=="Technical Degree"]=6
# Changing value of Gender into 1 and 2 format
hr$Gender[hr$Gender=="Female"]=1
hr$Gender[hr$Gender=="Male"]=2

#Changing the format of JobRole into 1,2,3 and 4  format
hr$JobRole[hr$JobRole=="Healthcare Representative"]=1
hr$JobRole[hr$JobRole=="Human Resources"]=2
hr$JobRole[hr$JobRole=="Laboratory Technician"]=3
hr$JobRole[hr$JobRole=="Manager"]=4
hr$JobRole[hr$JobRole=="Manufacturing Director"]=5
hr$JobRole[hr$JobRole=="Research Director"]=6
hr$JobRole[hr$JobRole=="Research Scientist"]=7
hr$JobRole[hr$JobRole=="Sales Executive"]=8
hr$JobRole[hr$JobRole=="Sales Representative"]=9
# Changing the format of MaritalStatus 1,2 and 3
hr$MaritalStatus[hr$MaritalStatus=="Divorced"]=1
hr$MaritalStatus[hr$MaritalStatus=="Married"]=2
hr$MaritalStatus[hr$MaritalStatus=="Single"]=3
# Changing the format of OverTIme into 0 and 1 format
hr$OverTime[hr$OverTime=="No"]=0
hr$OverTime[hr$OverTime=="Yes"]=1
head(hr,2)
#  EmployeeCount have to removed beacuse all is having 1 value
#  EmployeeNumber have to be removed because it is not impacting the result
#  StandardHours have to be removed beacuse in all rows having the same value.
View(hr)
hr=hr[ , -c(9,10,26)]
View(hr)
dim(hr)
d=c(2,4,6,11,17,18,21,25,28,29,30,31)
colname=c("Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate","Percent Salary Hike","Total Working Years","Years At Company","Years in Current Roles","Years Since Last Promotion","Years With Current Manager")

hrf=hr[,c(2,4,6,11,17,18,21,25,28,29,30,31)]
pl = colnames(hrf)
colname=c("Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate","Percent Salary Hike","Total Working Years","Years At Company","Years in Current Roles","Years Since Last Promotion","Years With Current Manager")
par(mfrow=c(2,1))
for (i in 1:length(pl)) {
  
  x <- hrf[,i]
  boxplot(x,horizontal = T,main = paste("Boxplot for", colname[i]))
  hist(x,main = paste("Histogram for", colname[i]))
}
# From the Box plot and the Histogram for many attributes some is not showing outliers and some is showing outliers due to do the mistach between the no. of experiece , Salary , position and many other factors are also associated with outliers which is valid .
# So we can't remove the outliers from the dataset as it hold true.
