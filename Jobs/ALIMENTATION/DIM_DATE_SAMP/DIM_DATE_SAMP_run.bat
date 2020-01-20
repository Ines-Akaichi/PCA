%~d0
cd %~dp0
java -Dtalend.component.manager.m2.repository=%cd%/../lib -Xms1024M -Xmx4096M -cp .;../lib/routines.jar;../lib/dom4j-1.6.1.jar;../lib/log4j-1.2.17.jar;../lib/postgresql-42.2.5.jar;../lib/talend_file_enhanced_20070724.jar;../lib/talendcsv.jar;dim_date_samp_0_1.jar; stagepca.dim_date_samp_0_1.DIM_DATE_SAMP  --context=Default %*