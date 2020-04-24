
# Set working directory
mydir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(mydir)

# Read data
mydata <- read.csv(paste0(mydir, '/Image_Analysis/data.csv'))

# Remove dust if necessary (outliers)
mydata2 <- mydata[which(mydata$Area > 1000), ]
mydata2$Accession <- as.character(mydata2$Image_Name)

mydata2$Ploidy <- substr(mydata2$Accession, nchar(mydata2$Accession)-10,
                         nchar(mydata2$Accession))

mydata2$Ploidy <- substr(mydata2$Accession, nchar(mydata2$Accession)-9,
                         nchar(mydata2$Accession)-8)

mydata2$Ploidy <-  as.factor(mydata2$Ploidy)


mydata2$Accession <- substr(mydata2$Accession, 1,nchar(mydata2$Accession)-11)
mydata2$Accession <- as.factor(mydata2$Accession)

# Convert px to mm
# Original Height = 2097 px (44mm)
# Original Width (or length) = 1624 px (34mm)
# 1200 dpi

px2mm <- 0.5 * ( (1624/34) + (2097/44) )    # Average of both dimensions

mydata2$Area <- mydata2$Area/ (34*44)
mydata2$Length <- mydata2$Length / px2mm
mydata2$Width <- mydata2$Width / px2mm


library(ggplot2)


mydata2$Accession <- factor(mydata2$Accession, 
                            levels = c("PI206268","PI422463","PI676177",
                                       "PI251123","PI251127","PI251128",
                                       "PI234896","PI234898","PI268234",
                                       "PI302899","PI311403","PI318989",
                                       "Quatro","Beacon"))

# Plots
bp1 <- ggplot(mydata2, aes(x=Accession, y=Area, fill = Ploidy)) +
  geom_boxplot() + 
  theme_classic() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs(y= expression(paste("Seed Area  ", (mm^2))), x = "Accession")

bp1





# eb1 <- ggplot(mydata2, aes(x=Accession, y=Area, fill = Ploidy)) + 
#   geom_bar(stat="identity", color="black", 
#            position=position_dodge()) 
# 
# eb1





# Stats
M1 <- lm(Area ~ Ploidy + Accession, data = mydata2)
# M2 <- lm(Area ~ Accession, data = mydata2)
summary(M1)
anova(M1)
summary(M2)
anova(M2)


# Remove controls
mydata3 <- mydata2[-which(mydata2$Accession == 'Quatro'), ]
mydata3 <- mydata3[-which(mydata3$Accession == 'Beacon'), ]




# Stats
M1 <- lm(Area ~ Ploidy + Accession, data = mydata3)
M2 <- lm(Area ~ Ploidy, data = mydata3)
summary(M1)
anova(M1)
summary(M2)
anova(M2)



# final anova table
M2 <- lm(Area ~ Ploidy, data = mydata3)
summary(M2)
anova(M2)

# Diagnostic plots
plot(M2)

