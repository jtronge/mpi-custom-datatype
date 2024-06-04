pdf("ddt_analysis.pdf");

library("ggplot2");
library("reshape");

data <- read.table("ddtbench.out.bz2", header=TRUE);

# exclude some tests 
data <- data[data$testname != "FFT2",];
data <- data[data$testname != "NAS_MG_x",];
data <- data[data$testname != "NAS_MG_y",];
data <- data[data$testname != "NAS_MG_z",];
data <- data[data$testname != "WRF_x_vec",];
data <- data[data$testname != "WRF_y_vec",];
data <- data[data$testname != "LAMMPS_atomic",];
data <- data[data$testname != "LAMMPS_full",];


# helper functions

si_num <- function (x) {

  if (!is.na(x)) {
    if (x > 1e6) { 
      chrs <- strsplit(format(x, scientific=12), split="")[[1]];
      rem <- chrs[seq(1,length(chrs)-6)];
      rem <- append(rem, "M");
    }
    
    else if (x > 1e3) { 
      chrs <- strsplit(format(x, scientific=12), split="")[[1]];
      rem <- chrs[seq(1,length(chrs)-3)];
      rem <- append(rem, "K");
    }
    else {
      return(x);
    }
    retval <- paste(rem, sep="", collapse="");
    return(retval);
  }
  else return(NA);
} 

si_vec <- function(x) {
  sapply(x, FUN=si_num);
}

my_labeller <- function(var, value){
  value <- as.character(value)
  if (var=="method") { 
    value[value=="manual"] <- "Manual packing"
    value[value=="mpi_ddt"]   <- "MPI DDTs"
  }
  return(value)
}


ndata <- cast(data=data, formula=testname+bytes ~ method+id, value="time", fun.aggregate=median);

  # packing = pack+unpack
  ndata$manual_packing = ndata$manual_pack + ndata$manual_unpack;
  
  # build a new dataframe, first data for manual packing, then mpi ddts
  testname <- as.character(ndata$testname);
  bytes <-ndata$bytes;
  value <- (ndata$bytes / ((ndata$manual_communication + ndata$manual_packing) / 2))
  pp_data <- aggregate(data.frame(ndata$bytes, ndata$reference_communication), by=list(Bytes = ndata$bytes), FUN=min)
  testname <- append(testname, rep("Traditional Ping-Pong", length(pp_data[,1])));
  bytes <- append(bytes, pp_data[,1]);
  value <- append(value, (pp_data[,1] / (pp_data[,3] / 2)));
  method <- rep("manual", length(bytes));
  
  testname <- append(testname, as.character(ndata$testname));
  bytes <- append(bytes, ndata$bytes);
  value <- append(value, ndata$bytes / (ndata$mpi_ddt_communication / 2));
  pp_data <- aggregate(data.frame(ndata$bytes, ndata$reference_communication), by=list(Bytes = ndata$bytes), FUN=min)
  testname <- append(testname, rep("Traditional Ping-Pong", length(pp_data[,1])));
  bytes <- append(bytes, pp_data[,1]);
  value <- append(value, (pp_data[,1] / (pp_data[,3] / 2)));
  method <- append(method, rep("mpi_ddt", length(ndata$bytes) + length(pp_data[,1])));
  
  newdf <- data.frame(testname, method, bytes, value);
  
  m <- ggplot(newdf, aes(x=bytes, y=value, color=testname, shape=testname)) +
    facet_grid(. ~ method, scales="free", labeller=my_labeller) +
    geom_line() +
    geom_point(size=4) +
    scale_x_continuous('Datasize [Byte]', labels=si_vec) + 
    scale_y_continuous('Bandwidth [MB/s]') +
    scale_colour_discrete(name="Test Name") + 
    scale_shape_manual(name="Test Name", values=seq(1, length(unique(ndata$testname))+1)) +
    theme(axis.text.x = element_text(angle=90));
  
  print(m);
  rm(testname, bytes, value, method, pp_data, newdf, m)

  # create new dataframe [testname  method  bytes  id   time] but
  # for the manual method, the communication id is the sum of comm + pack + unpack

  ndata <- cast(data=data, formula=testname+bytes+method ~ id, value="time", fun.aggregate=median);

  # we can do this for all methods, because for reference and mpi_ddt pack/unpack is 0.
  ndata$communication <- ndata$communication + ndata$pack + ndata$unpack;

  mndata <- melt(ndata, id=c("testname", "method", "bytes"));

  # do not plot reference and mpi_pack_ddt
  mndata <- mndata[mndata$method != "reference", ];
  mndata <- mndata[mndata$method != "mpi_pack_ddt", ];

  # remove pack/unpack for mpi_ddt method, because it us 0 anyway
  mndata <- mndata[(mndata$method != "mpi_ddt") | (mndata$id != "pack"),];  
  mndata <- mndata[(mndata$method != "mpi_ddt") | (mndata$id != "unpack"),];

  makeplot <- function(name) {
    print(name);
    m <- ggplot(mndata[mndata$testname == name,], aes(x=bytes, y=value, color=id, shape=id)) +
      facet_grid(. ~ method) +
      geom_line() +
      geom_point(size=4) +
      labs(title=name) +
      scale_x_continuous('Datasize [Byte]', labels=si_vec) + 
      scale_y_continuous('Time [us]') + 
      scale_shape_manual(name="Phase", breaks=c("communication", "ddt_create_overhead", "ddt_free_overhead", "pack", "unpack"), 
                         labels=c("Communication RTT\n (incl. Packing, excl. DDT create)", "DDT Create", "DDT Free", "Pack", "Unpack"),
                         values=c(21,22,23,24,25)) +
      scale_color_manual(name="Phase", breaks=c("communication", "ddt_create_overhead", "ddt_free_overhead", "pack", "unpack"), 
                         labels=c("Communication RTT\n (incl. Packing, excl. DDT create)", "DDT Create", "DDT Free", "Pack", "Unpack"),
                         values=c(2:6)) +
      theme(axis.text.x = element_text(angle=90));
    print(m);
    return(m);
  }

  plots = lapply(as.character(unique(mndata$testname)), makeplot);

dev.off()
