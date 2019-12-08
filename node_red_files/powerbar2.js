<script>
(function($scope) {
    
let amount = 10 //amount of LEDs
let label = "LED bar Graph"
 
let on1 = "#00FF00"
let off1 = "#006600"
let on2 = "#FFFF00"
let off2 = "#666600"
let on3 = "#FF0000"
let off3 = "#660000"

let threshold1 = 1/2 // the limit between color 1 and color 2
let threshold2 = 4/5 //  the limit between color 2 and color 3

var bargraph = new Array(amount).fill("#000000")

$scope.$watch('msg', function() {
    
if ($scope.msg){
    if ($scope.msg.hasOwnProperty('payload') && typeof $scope.msg.payload == "number"){
        $scope.msg.label = label
        $scope.msg.payload = parseInt($scope.msg.payload)
    
        if ($scope.msg.payload > amount){
            $scope.msg.payload = amount
        }
        for (var i = 0; i < $scope.msg.payload; i++){
            if (i < amount*threshold1){
                bargraph[i] = on1
            }else if (i < amount*threshold2){
                bargraph[i] = on2
            }else{
                bargraph[i] = on3
            }
        }
        for (var i = $scope.msg.payload; i < amount; i++){
            if (i < amount*threshold1){
                bargraph[i] = off1
            }else if (i < amount*threshold2){
                bargraph[i] = off2
            }else{
                bargraph[i] = off3
            }
        }
        $scope.msg.bargraph = bargraph.reverse()
    }else if (typeof $scope.msg.payload !== "number"){
        $scope.msg = {"bargraph":[...bargraph], "payload": 0, "label":"Led Bar Graph"}
    }   
}else{
    $scope.msg = {"bargraph":[...bargraph], "payload": 0, "label":"Led Bar Graph"}
}
    });
})(scope);
</script>

<style>
.bargraph {
    float: right;
    padding: 3px;
    width: 3px;
    height: 10px;
    margin: 4px 2px 8px 0px;
    border-radius: 0%;
}
</style>

<div>{{msg.label}}
<span ng-repeat="led in msg.bargraph track by $index">
    <span class="bargraph" style="background-color: {{led}}; box-shadow: black 0 -1px 1px 0px, inset black  0 -1px 4px, {{led}} 0 3px 15px;"></span>
</span>
</div>
