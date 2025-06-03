#!/bin/bash
export AIRFLOW_HOME=$(pwd)

airflow db reset
airflow standalone
