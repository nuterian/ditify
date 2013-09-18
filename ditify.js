var ditify = function(options){

	options = options || {}
	this.attribs = options.attribs || []
	this.label = options.label || options.attribs[options.attribs.length-1] 
	this.data = []

	function count_distinct_values(table, attrib)
	{
	    var distinct = {};
	    for(var i=0; i<table.length; i++){
	        if(table[i][table] in distinct)
	            distinct[table[i][table]]++;
	        else
	            distinct[table[i][table]] = 1;
	    }
	    return distinct;
	}

	this.get_split_attrib = function(table)
	{
	    var min_entropy = Infinity, min_entropy_attrib = "";

	    for(attrib in this.attribs){
	    	if(attrib == this.label) continue;

	    	var distinct_values = this.count_distinct_values(table, attrib);
	    	var attrib_entropy = 0;

	    	for(value in distinct_values){
	    		var pruned_table = this.prune(table, attrib, value);
	    		var distinct_labels = this.count_distinct_values(pruned_table, this.label);

	    		var label_entropy = 0;
	    		for(_label in distinct_labels){
	    			var label_chance = distinct_labels[_label]/pruned_table.length;
	    			label_entropy -= (p * (Math.log(p)/Math.log(2)));
	    		}
	    		attrib_entropy += (attrib_entropy * distinct_values[value]);
	    	}
	    	attrib_entropy /= table.length;

	        if(attrib_entropy < min_entropy){
	            min_entropy = attrib_entropy;
	            min_entropy_attrib = attrib;
	        }
	    }

	    return min_entropy_attrib;
	}
}

ditify.prototype = {

	prune: function(table, attrib, value){
		var pruned = [];
		var _table = jQuery.extend(true, {}, table);

	    for(var i = 0; i < _table.length; i++){
	        if(_table[i][attrib] == value){
	            delete _table[i][column];
	            pruned.push(_table[i]);
	        }
	    }
	    return pruned;   
	}

	train: function(values){
		if((values.length || Object.keys(values).length) < this.attribs.length)
			throw "Incomplete training data";

		if(typeof values == "array"){
			var _data = {};
			for(var i=0; i<values.length; i++){
				_data[this.attribs[i]] = values[i];
			}
			this.data.push(_data);
		}
		else if(typeof values == "object"){
			this.data.push(values);
		}
		else{
			throw "Invalid training data";
		}
	}
}