$fileDir = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $fileDir
java '-Dtalend.component.manager.m2.repository=%cd%/../lib' '-Xms1024M' '-Xmx4096M' -cp '.;../lib/routines.jar;../lib/advancedPersistentLookupLib-1.2.jar;../lib/commons-collections-3.2.2.jar;../lib/commons-collections4-4.1.jar;../lib/dom4j-1.6.1.jar;../lib/geronimo-stax-api_1.0_spec-1.0.1.jar;../lib/jboss-serialization.jar;../lib/log4j-1.2.17.jar;../lib/poi-3.16-20170419_modified_talend.jar;../lib/poi-ooxml-3.16-20170419_modified_talend.jar;../lib/poi-ooxml-schemas-3.16-20170419.jar;../lib/poi-scratchpad-3.16-20170419.jar;../lib/postgresql-42.2.5.jar;../lib/trove.jar;../lib/xmlbeans-2.6.0.jar;ft_biological_measures_0_1.jar;' stagepca.ft_biological_measures_0_1.FT_BIOLOGICAL_MEASURES  --context=Default %*