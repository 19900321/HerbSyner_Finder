## the original script attempted to change working directory to a hardcoded path
## in the web app we run the script using subprocess with cwd set to the job directory,
## so there's no need to change it here. comment out the setwd call.
# setwd("XX/code")
# getwd()

library(pheatmap)
library(showtext)
library(RColorBrewer)
library(dplyr)

font_add('Arial','/Library/Fonts/Arial.ttf') 
showtext_auto()
matrix_cluster<- read.csv('result/matrix_cluster_result.csv',check.names=F)
matrix_type<- read.csv('result/matrix_type_result.csv',check.names=F)


# unique node labels (ingredients/herbs) and combination names (FJ) come from the type file
unique_values <- unique(matrix_type[, 1])
new_data <- data.frame(node = unique_values)
# derive combination names dynamically instead of hardcoding
columns <- unique(matrix_type[, 2])
for (column in columns) {
  new_column <- paste(column)
  new_data[, new_column] <- 0
  for (i in seq_along(unique_values)) {
    has_value <- column %in% matrix_type[matrix_type[, 1] == unique_values[i], 2]
    if (has_value) {
      new_data[i, new_column] <- 1
    }
  }
}

rownames(matrix_cluster)=matrix_cluster[,1]
matrix_cluster=matrix_cluster[,-1]
# build annotation data frame dynamically from the computed column names
annotation_col <- data.frame(row.names = new_data$node)
for (col in columns) {
  annotation_col[[col]] <- new_data[[col]]
}

# create a uniform color mapping for each combination name via loop
ann_colors <- list()
for (col in columns) {
  ann_colors[[col]] <- c("#F7F7F7", '#41AB5D')
}

plot=pheatmap(matrix_cluster,cutree_rows = 3,
              cutree_cols = 3,
              colorRampPalette(rev(brewer.pal(11,"RdBu")[2:6]))(100),
              treeheight_col = 10,treeheight_row = 20,fontsize=14,
              annotation_col=annotation_col,
              annotation_colors = ann_colors,
              display_numbers = FALSE,
              angle_col = 45)

outputFolder <- file.path("result/figure")
pdf(file = file.path(outputFolder, "heatmap.pdf"), width =20, height = 10)

plot
dev.off()


#annotation
matrix_herb_type<- read.csv('result/matrix_herb_result.csv',check.names=F)
matrix_data <- as.matrix(table(matrix_herb_type$herb, matrix_herb_type$ingredient) > 0)
numeric_matrix <- apply(matrix_data , c(1, 2), function(x) ifelse(x, 1, 0))
numeric_matrix <- as.data.frame(numeric_matrix)
transposed_matrix <- t(numeric_matrix)

anno=pheatmap(transposed_matrix,
              color = c("#F7F7F7", '#A6D96A'),
              treeheight_col = 20,
              treeheight_row = 20,
              fontsize = 14,
              angle_col = 90)

anno
outputFolder <- file.path( "result/figure")
pdf(file = file.path(outputFolder, "Figure6_annatation_cluster_ingre_fangji_heatmap——2.pdf"), width =10, height = 6)
anno
dev.off()



