<?php
include "./class/database.php";
include "./class/city.php";
$city=new city();
$city->id=$_GET["id"];
if($city->delete())
{
    echo 1;
}
else
{
    echo 0;
}
?>